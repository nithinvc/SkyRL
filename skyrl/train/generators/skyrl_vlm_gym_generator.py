"""
SkyRLVLMGymGenerator: VLM (vision-language model) multi-turn RL generator.

Subclasses SkyRLGymGenerator to handle multi-modal observations (images)
from VisGym environments. Uses pure Python types (OpenAI-format messages
with base64-encoded images) rather than Tinker chunks.

Token bookkeeping uses a "render delta" approach: the conversation (list of
messages) is the source of truth and is re-tokenized via vLLM's
render_chat_completion at each step.  Generated tokens keep their original
logprobs; observation tokens are obtained by slicing the re-render and
are masked out (loss_mask=0).
"""

import asyncio
import copy
import json

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from uuid import uuid4

import numpy as np
import torch
from loguru import logger
from tqdm.asyncio import tqdm

import skyrl_gym
from skyrl.backends.skyrl_train.inference_engines.base import (
    ConversationType,
    InferenceEngineInput,
    MultiModalFeatures,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.env_vars import SKYRL_GENERATE_CONCURRENCY_PER_ENGINE
from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.base import GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl.train.generators.skyrl_gym_generator import (
    SkyRLGymGenerator,
    TrajectoryOutput,
    TurnProfile,
)
from skyrl.train.generators.utils import (
    apply_overlong_filtering,
    get_rollout_metrics,
)


class RenderedConversation(TypedDict):
    prompt_ids: list[int]
    features: MultiModalFeatures


def deserialize_mm_features(features: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deserialize multimodal features from a render_chat_completion response.

    Decodes base64-encoded vision tensors from the ``kwargs_data`` field
    returned by vLLM's ``/v1/chat/completions/render`` endpoint.

    Args:
        features: The ``features`` dict from a render response.

    Returns:
        ``(pixel_values, image_grid_thw)`` — ``pixel_values`` concatenated along the patch
        dimension; ``image_grid_thw`` stacked to ``[num_images, 3]`` (one row per image).
        Returns empty tensors when no vision data is present.
    """
    kwargs_data = (features or {}).get("kwargs_data")
    if not kwargs_data or "image" not in kwargs_data:
        return torch.empty(0), torch.empty(0, 3, dtype=torch.long)

    from vllm.entrypoints.serve.disagg.mm_serde import (
        decode_mm_kwargs_item as _vllm_decode,
    )

    pv_parts: list[torch.Tensor] = []
    thw_parts: list[torch.Tensor] = []
    for b64_str in kwargs_data["image"]:
        if b64_str is None:
            continue  # cached item — tensor data not included
        item = _vllm_decode(b64_str)
        data = item.get_data()
        if "pixel_values" in data and isinstance(data["pixel_values"], torch.Tensor):
            pv_parts.append(data["pixel_values"])
        if "image_grid_thw" in data and isinstance(data["image_grid_thw"], torch.Tensor):
            thw_parts.append(data["image_grid_thw"])

    pixel_values = torch.cat(pv_parts, dim=0) if pv_parts else torch.empty(0)
    image_grid_thw = torch.stack(thw_parts, dim=0) if thw_parts else torch.empty(0, 3, dtype=torch.long)
    return pixel_values, image_grid_thw


class SkyRLVLMGymGenerator(SkyRLGymGenerator):
    """VLM generator that handles multi-modal (text + image) observations.

    Simplifies the parent SkyRLGymGenerator by dropping support for:
    - custom_chat_template / retokenize_chat_history
    - step_wise_trajectories
    - batched generation
    - expert indices tracking

    Inherits the parent's generate() method which dispatches to agent_loop()
    in parallel via asyncio.
    """

    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        skyrl_gym_cfg: SkyRLGymConfig,
        inference_engine_client: RemoteInferenceClient,
        tokenizer,
    ):
        # Parent stores as self.inference_engine_client and sets up
        # generator_cfg, skyrl_gym_cfg, tokenizer, max_turns, env_executor, etc.
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer)
        self.inference_client = inference_engine_client
        logger.info("Initialized SkyRLVLMGymGenerator (VLM multi-modal generator)")

    def _validate_cfg(self, generator_cfg: GeneratorConfig):
        if generator_cfg.batched:
            raise ValueError("SkyRLVLMGymGenerator does not support batched generation. Set `batched=False`.")
        if generator_cfg.step_wise_trajectories:
            raise ValueError("SkyRLVLMGymGenerator does not support step-wise trajectories.")
        if not generator_cfg.use_conversation_multi_turn:
            raise ValueError(
                "SkyRLVLMGymGenerator requires `use_conversation_multi_turn=True` "
                "because multi-modal observations must be in separate user messages."
            )

    async def _render_conversation(self, conversation: ConversationType) -> RenderedConversation:
        rendered = await self.inference_client.render_chat_completion(
            {"json": {"model": self.inference_client.model_name, "messages": conversation}}
        )
        return RenderedConversation(prompt_ids=rendered["token_ids"], features=rendered.get("features", {}))

    async def agent_loop(
        self,
        prompt: ConversationType,
        env_class: str,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        trajectory_id: Optional[TrajectoryID] = None,
    ) -> TrajectoryOutput:
        """Multi-turn VLM generation loop for a single trajectory.

        Uses the conversation as the source of truth and re-tokenizes via
        vLLM's render_chat_completion at each step (render delta approach).
        Generated tokens keep their original logprobs; observation tokens
        are obtained by slicing the re-render and masked out (loss_mask=0).
        """
        # ── Setup ──────────────────────────────────────────────────────
        env_extras["max_turns"] = self.max_turns
        env_config = getattr(self.skyrl_gym_cfg, env_class, dict())
        env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extras)

        session_id = (
            f"{trajectory_id.instance_id}_{trajectory_id.repetition_id}" if trajectory_id is not None else uuid4().hex
        )

        conversation = copy.deepcopy(prompt)
        conversation, _ = await self._run_in_executor_if_available(env.init, conversation)

        # Render initial conversation → prompt_ids
        # latest_features always points to the most recent render's features
        # (each render covers the full conversation, so later renders supersede earlier ones)
        initial_render = await self._render_conversation(conversation)
        prompt_ids = initial_render["prompt_ids"]
        latest_features = initial_render["features"]

        current_sampling_params: dict = (
            sampling_params if sampling_params is not None else asdict(self.generator_cfg.sampling_params)
        )
        get_logprobs = self.generator_cfg.sampling_params.logprobs is not None

        # ── Accumulators ───────────────────────────────────────────────
        response_ids: List[int] = []
        loss_mask: List[int] = []
        rollout_logprobs: Optional[List[float]] = [] if get_logprobs else None
        per_step_rewards: List[Tuple[float, int]] = []
        stop_reason = "stop"
        done = False

        # ── Main loop ─────────────────────────────────────────────────
        # To avoid a second render call per turn, we defer obs-token
        # extraction: after appending obs we just record the slice offset,
        # then compute the actual obs tokens from the *next* turn's render
        # (which produces identical token_ids since the conversation hasn't
        # changed in between).
        pending_obs_offset: Optional[int] = None
        turn_profiles: List[TurnProfile] = []
        turn_idx = 0

        while not done:
            # 1. Render full conversation for this turn's generation input
            t0 = time.perf_counter()
            rendered_conversation = await self._render_conversation(conversation)
            render_time = time.perf_counter() - t0

            input_ids = rendered_conversation["prompt_ids"]
            latest_features = rendered_conversation["features"]

            # 1b. Flush pending obs tokens from the previous turn
            if pending_obs_offset is not None:
                obs_tokens = input_ids[pending_obs_offset:]
                response_ids.extend(obs_tokens)
                loss_mask.extend([0] * len(obs_tokens))
                if rollout_logprobs is not None:
                    rollout_logprobs.extend([0.0] * len(obs_tokens))
                pending_obs_offset = None

            if len(input_ids) > max_input_length:
                stop_reason = "length"
                break

            # 2. Generate
            engine_input = InferenceEngineInput(
                prompt_token_ids=[input_ids],
                session_ids=[session_id],
                sampling_params=current_sampling_params,
                mm_features=latest_features,
            )
            t0 = time.perf_counter()
            engine_output = await self.inference_client.generate(engine_input)
            generate_time = time.perf_counter() - t0

            gen_text = engine_output["responses"][0]
            gen_ids = engine_output["response_ids"][0]
            stop_reason = engine_output["stop_reasons"][0]
            gen_logprobs = engine_output["response_logprobs"][0] if engine_output.get("response_logprobs") else None

            # 3. Environment step
            t0 = time.perf_counter()
            env_step_output = await self._run_in_executor_if_available(env.step, gen_text)
            env_step_time = time.perf_counter() - t0

            new_obs = env_step_output["observations"]
            step_reward: float = env_step_output["reward"]
            done = env_step_output["done"]

            turn_profiles.append(TurnProfile(
                turn=turn_idx,
                render_time_s=render_time,
                generate_time_s=generate_time,
                env_step_time_s=env_step_time,
                deserialize_time_s=0.0,
                num_input_tokens=len(input_ids),
                num_output_tokens=len(gen_ids),
            ))
            turn_idx += 1

            # 4. Append assistant message to conversation
            conversation.append({"role": "assistant", "content": gen_text})

            # 5. Track generated tokens (loss_mask=1)
            response_ids.extend(gen_ids)
            loss_mask.extend([1] * len(gen_ids))
            if rollout_logprobs is not None:
                rollout_logprobs.extend(gen_logprobs if gen_logprobs else [0.0] * len(gen_ids))

            per_step_rewards.append((step_reward, len(response_ids) - 1))

            # 6. If episode continues, defer obs token extraction to next render
            if not done:
                conversation.extend(new_obs)
                pending_obs_offset = len(input_ids) + len(gen_ids)

        # ── Build per-token rewards ───────────────────────────────────
        per_token_reward: List[float] = [0.0] * len(response_ids)
        for reward, idx in per_step_rewards:
            per_token_reward[idx] = float(reward)

        # ── Deserialize vision tensors from the most recent render ────
        t0 = time.perf_counter()
        pixel_values, image_grid_thw = deserialize_mm_features(latest_features)
        deserialize_time = time.perf_counter() - t0

        if turn_profiles:
            last = turn_profiles[-1]
            turn_profiles[-1] = TurnProfile(
                turn=last.turn,
                render_time_s=last.render_time_s,
                generate_time_s=last.generate_time_s,
                env_step_time_s=last.env_step_time_s,
                deserialize_time_s=deserialize_time,
                num_input_tokens=last.num_input_tokens,
                num_output_tokens=last.num_output_tokens,
            )

        # ── Cleanup ───────────────────────────────────────────────────
        env_metrics = env.get_metrics()
        await self._run_in_executor_if_available(env.close)

        return TrajectoryOutput(
            response_ids=response_ids,
            reward=per_token_reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            rollout_logprobs=rollout_logprobs,
            env_metrics=env_metrics,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            conversation=conversation,
            turn_profiles=turn_profiles,
        )

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        """Override parent generate() to collect turn profiles and export a profiling report.

        Uses a semaphore to bound the number of concurrently active agent_loops,
        backfilling immediately as each trajectory completes.
        """
        prompts = input_batch["prompts"]
        env_classes = input_batch["env_classes"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch.get("trajectory_ids", None)
        sampling_params: Optional[dict] = input_batch.get("sampling_params", None)
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length

        # ── Bounded concurrency ───────────────────────────────────────
        k = self.generator_cfg.max_concurrent_trajectories
        if k is None:
            ie_cfg = self.generator_cfg.inference_engine
            num_engines = (
                len(ie_cfg.external_server_urls)
                if ie_cfg.external_server_urls
                else ie_cfg.num_engines
            )
            k = SKYRL_GENERATE_CONCURRENCY_PER_ENGINE * num_engines
        sem = asyncio.Semaphore(k) if k > 0 else None
        logger.info(
            f"Bounded concurrency: max_concurrent_trajectories={k} "
            f"(0=unlimited), total trajectories={len(prompts)}"
        )

        async def _bounded_agent_loop(idx: int) -> TrajectoryOutput:
            queued_at = time.perf_counter()
            if sem is not None:
                await sem.acquire()
            queue_wait = time.perf_counter() - queued_at
            try:
                output = await self.agent_loop(
                    prompts[idx],
                    env_classes[idx],
                    env_extras[idx],
                    max_tokens,
                    max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[idx] if trajectory_ids is not None else None,
                )
            finally:
                if sem is not None:
                    sem.release()
            if output.turn_profiles:
                output.turn_profiles[0].queue_wait_time_s = queue_wait
            return output

        tasks = [_bounded_agent_loop(i) for i in range(len(prompts))]
        wall_clock_start = time.perf_counter()
        all_outputs: List[TrajectoryOutput] = await tqdm.gather(
            *tasks,
            desc="Generating Trajectories",
            miniters=max(1, len(tasks) // 10),
            mininterval=5,
            disable=disable_tqdm,
        )
        wall_clock_s = time.perf_counter() - wall_clock_start

        # ── Collect and export profiling data ─────────────────────────
        all_profiles = [o.turn_profiles for o in all_outputs if o.turn_profiles]
        if all_profiles:
            self._export_profiling_report(all_profiles, wall_clock_s=wall_clock_s)

        # ── Aggregate outputs (non-step-wise VLM path) ────────────────
        responses = [output.response_ids for output in all_outputs]
        rewards = [output.reward for output in all_outputs]
        stop_reasons = [output.stop_reason for output in all_outputs]
        loss_masks = [output.loss_mask for output in all_outputs]
        prompt_token_ids = [output.prompt_ids for output in all_outputs]
        env_metrics = [output.env_metrics for output in all_outputs]

        has_pixel_values = any(output.pixel_values is not None for output in all_outputs)
        pixel_values = [output.pixel_values for output in all_outputs] if has_pixel_values else None
        image_grid_thw = [output.image_grid_thw for output in all_outputs] if has_pixel_values else None

        has_conversations = any(output.conversation is not None for output in all_outputs)
        conversations = [output.conversation for output in all_outputs] if has_conversations else None

        if sampling_params is not None:
            get_logprobs = sampling_params.get("logprobs", None) is not None
        else:
            get_logprobs = self.generator_cfg.sampling_params.logprobs is not None

        rollout_logprobs = (
            [output.rollout_logprobs for output in all_outputs] if get_logprobs else None
        )

        if self.generator_cfg.inference_engine.enable_return_routed_experts:
            rollout_expert_indices = [output.rollout_expert_indices for output in all_outputs]
        else:
            rollout_expert_indices = None

        rollout_metrics = get_rollout_metrics(responses, rewards, env_metrics, env_classes)

        if self.generator_cfg.zero_reward_on_non_stop:
            rewards = self._zero_reward_if_not_stop(rewards, stop_reasons)

        if self.generator_cfg.apply_overlong_filtering:
            loss_masks = apply_overlong_filtering(loss_masks, stop_reasons)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": rollout_logprobs,
            "trajectory_ids": None,
            "rollout_expert_indices": rollout_expert_indices,
            "is_last_step": None,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "conversations": conversations,
        }

        return generator_output

    def _export_profiling_report(self, all_profiles: List[List[TurnProfile]], *, wall_clock_s: float = 0.0) -> None:
        """Export raw profiles, aggregated summary, and graphs to the experiment directory."""
        exp_name = self.generator_cfg.exp_name
        exp_dir = Path("/home/ray/default/skyrl-vlm/SkyRL/inference-exps") / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # ── Raw data ──────────────────────────────────────────────────
        raw = [[asdict(tp) for tp in traj] for traj in all_profiles]
        (exp_dir / "raw_profiles.json").write_text(json.dumps(raw, indent=2))

        # ── Aggregate by turn number ─────────────────────────────────
        turn_data: Dict[int, Dict[str, list]] = {}
        fields = [
            "render_time_s", "generate_time_s", "env_step_time_s",
            "deserialize_time_s", "num_input_tokens", "num_output_tokens",
            "queue_wait_time_s",
        ]
        for traj in all_profiles:
            for tp in traj:
                bucket = turn_data.setdefault(tp.turn, {f: [] for f in fields})
                for f in fields:
                    bucket[f].append(getattr(tp, f))

        def _stats(values: list) -> dict:
            a = np.array(values, dtype=np.float64)
            return {
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "p50": float(np.median(a)),
                "p95": float(np.percentile(a, 95)),
                "min": float(np.min(a)),
                "max": float(np.max(a)),
                "count": len(values),
            }

        summary: Dict[str, Any] = {"per_turn": {}, "totals": {}}
        for turn_num in sorted(turn_data):
            summary["per_turn"][turn_num] = {
                f: _stats(turn_data[turn_num][f]) for f in fields
            }
            gen_times = np.array(turn_data[turn_num]["generate_time_s"])
            out_tokens = np.array(turn_data[turn_num]["num_output_tokens"], dtype=np.float64)
            in_tokens = np.array(turn_data[turn_num]["num_input_tokens"], dtype=np.float64)
            safe_out = np.where(out_tokens > 0, out_tokens, 1.0)
            safe_in = np.where(in_tokens > 0, in_tokens, 1.0)
            summary["per_turn"][turn_num]["generate_time_per_output_token_s"] = _stats(
                (gen_times / safe_out).tolist()
            )
            summary["per_turn"][turn_num]["generate_time_per_input_token_s"] = _stats(
                (gen_times / safe_in).tolist()
            )

        total_times = {f: 0.0 for f in fields[:4]}
        total_input_tokens = 0
        total_output_tokens = 0
        queue_waits: list = []
        for traj in all_profiles:
            if traj:
                queue_waits.append(traj[0].queue_wait_time_s)
            for tp in traj:
                total_times["render_time_s"] += tp.render_time_s
                total_times["generate_time_s"] += tp.generate_time_s
                total_times["env_step_time_s"] += tp.env_step_time_s
                total_times["deserialize_time_s"] += tp.deserialize_time_s
                total_input_tokens += tp.num_input_tokens
                total_output_tokens += tp.num_output_tokens

        summary["totals"] = {
            **total_times,
            "total_time_s": sum(total_times.values()),
            "num_trajectories": len(all_profiles),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "queue_wait_time_s": _stats(queue_waits) if queue_waits else None,
        }

        if wall_clock_s > 0:
            summary["rollout"] = {
                "wall_clock_s": wall_clock_s,
                "num_trajectories": len(all_profiles),
                "total_output_tokens": total_output_tokens,
                "total_input_tokens": total_input_tokens,
                "output_tokens_per_second": total_output_tokens / wall_clock_s,
                "input_tokens_per_second": total_input_tokens / wall_clock_s,
                "trajectories_per_second": len(all_profiles) / wall_clock_s,
                "seconds_per_output_token": wall_clock_s / max(total_output_tokens, 1),
            }

        (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        logger.info(f"Profiling report written to {exp_dir}")

        # ── Graphs ────────────────────────────────────────────────────
        try:
            self._generate_profiling_graphs(turn_data, total_times, all_profiles, exp_dir)
        except Exception:
            logger.exception("Failed to generate profiling graphs")

    def _generate_profiling_graphs(
        self,
        turn_data: Dict[int, Dict[str, list]],
        total_times: Dict[str, float],
        all_profiles: List[List[TurnProfile]],
        exp_dir: Path,
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        turns = sorted(turn_data.keys())
        component_keys = ["render_time_s", "generate_time_s", "env_step_time_s", "deserialize_time_s"]
        component_labels = ["Render", "Generate", "Env Step", "Deserialize"]

        # 1. Stacked bar: mean time per component by turn
        fig, ax = plt.subplots(figsize=(10, 6))
        bottoms = np.zeros(len(turns))
        for key, label in zip(component_keys, component_labels):
            means = [np.mean(turn_data[t][key]) for t in turns]
            ax.bar(turns, means, bottom=bottoms, label=label)
            bottoms += np.array(means)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean Time (s)")
        ax.set_title("Time per Component by Turn")
        ax.legend()
        ax.set_xticks(turns)
        fig.tight_layout()
        fig.savefig(exp_dir / "time_by_turn_stacked_bar.png", dpi=150)
        plt.close(fig)

        # 2. Line chart: mean input/output tokens by turn
        fig, ax = plt.subplots(figsize=(10, 6))
        mean_in = [np.mean(turn_data[t]["num_input_tokens"]) for t in turns]
        mean_out = [np.mean(turn_data[t]["num_output_tokens"]) for t in turns]
        ax.plot(turns, mean_in, marker="o", label="Input Tokens")
        ax.plot(turns, mean_out, marker="s", label="Output Tokens")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean Token Count")
        ax.set_title("Tokens by Turn")
        ax.legend()
        ax.set_xticks(turns)
        fig.tight_layout()
        fig.savefig(exp_dir / "tokens_by_turn.png", dpi=150)
        plt.close(fig)

        # 3. Line chart: generate time per output token by turn
        fig, ax = plt.subplots(figsize=(10, 6))
        time_per_out = []
        for t in turns:
            gen_t = np.array(turn_data[t]["generate_time_s"])
            out_t = np.array(turn_data[t]["num_output_tokens"], dtype=np.float64)
            safe = np.where(out_t > 0, out_t, 1.0)
            time_per_out.append(float(np.mean(gen_t / safe)))
        ax.plot(turns, time_per_out, marker="o", color="tab:red")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean Generate Time / Output Token (s)")
        ax.set_title("Generate Time per Output Token by Turn")
        ax.set_xticks(turns)
        fig.tight_layout()
        fig.savefig(exp_dir / "generate_time_per_output_token_by_turn.png", dpi=150)
        plt.close(fig)

        # 4. Pie chart: total time breakdown
        fig, ax = plt.subplots(figsize=(8, 8))
        sizes = [total_times[k] for k in component_keys]
        nonzero = [(s, l) for s, l in zip(sizes, component_labels) if s > 0]
        if nonzero:
            ax.pie(
                [s for s, _ in nonzero],
                labels=[l for _, l in nonzero],
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.set_title("Total Time Breakdown")
        fig.tight_layout()
        fig.savefig(exp_dir / "total_time_breakdown_pie.png", dpi=150)
        plt.close(fig)

        # 5. Histogram: generate latency distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        all_gen_times = []
        for traj in all_profiles:
            for tp in traj:
                all_gen_times.append(tp.generate_time_s)
        ax.hist(all_gen_times, bins=min(50, max(10, len(all_gen_times) // 5)), edgecolor="black")
        ax.set_xlabel("Generate Latency (s)")
        ax.set_ylabel("Count")
        ax.set_title("Generate Latency Distribution")
        fig.tight_layout()
        fig.savefig(exp_dir / "generate_latency_histogram.png", dpi=150)
        plt.close(fig)

        logger.info(f"Profiling graphs saved to {exp_dir}")

    async def generate_batched(self, *args, **kwargs) -> GeneratorOutput:
        raise NotImplementedError(
            "SkyRLVLMGymGenerator does not support batched generation. "
            "Use the default async agent_loop path instead."
        )
