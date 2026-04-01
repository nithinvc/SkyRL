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

import copy
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from uuid import uuid4

import torch
from loguru import logger

import skyrl_gym
from skyrl.backends.skyrl_train.inference_engines.base import (
    ConversationType,
    InferenceEngineInput,
    MultiModalFeatures,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.base import GeneratorOutput, TrajectoryID
from skyrl.train.generators.skyrl_gym_generator import (
    SkyRLGymGenerator,
    TrajectoryOutput,
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
        ``(pixel_values, image_grid_thw)`` — concatenated across all images.
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
    image_grid_thw = torch.cat(thw_parts, dim=0) if thw_parts else torch.empty(0, 3, dtype=torch.long)
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
        while not done:
            # 1. Render full conversation for this turn's generation input
            rendered_conversation = await self._render_conversation(conversation)
            input_ids = rendered_conversation["prompt_ids"]
            latest_features = rendered_conversation["features"]

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
            engine_output = await self.inference_client.generate(engine_input)

            gen_text = engine_output["responses"][0]
            gen_ids = engine_output["response_ids"][0]
            stop_reason = engine_output["stop_reasons"][0]
            gen_logprobs = engine_output["response_logprobs"][0] if engine_output.get("response_logprobs") else None

            # 3. Environment step
            env_step_output = await self._run_in_executor_if_available(env.step, gen_text)
            new_obs = env_step_output["observations"]
            step_reward: float = env_step_output["reward"]
            done = env_step_output["done"]

            # 4. Append assistant message to conversation
            conversation.append({"role": "assistant", "content": gen_text})

            # 5. Track generated tokens (loss_mask=1)
            response_ids.extend(gen_ids)
            loss_mask.extend([1] * len(gen_ids))
            if rollout_logprobs is not None:
                rollout_logprobs.extend(gen_logprobs if gen_logprobs else [0.0] * len(gen_ids))

            per_step_rewards.append((step_reward, len(response_ids) - 1))

            # 6. If episode continues, track observation tokens (loss_mask=0)
            if not done:
                conversation.extend(new_obs)

                # Render delta: re-render full conversation, slice off the new obs tokens
                obs_render = await self._render_conversation(conversation)
                full_ids = obs_render["prompt_ids"]
                latest_features = obs_render["features"]
                obs_tokens = full_ids[len(input_ids) + len(gen_ids) :]

                response_ids.extend(obs_tokens)
                loss_mask.extend([0] * len(obs_tokens))
                if rollout_logprobs is not None:
                    rollout_logprobs.extend([0.0] * len(obs_tokens))

        # ── Build per-token rewards ───────────────────────────────────
        per_token_reward: List[float] = [0.0] * len(response_ids)
        for reward, idx in per_step_rewards:
            per_token_reward[idx] = float(reward)

        # ── Deserialize vision tensors from the most recent render ────
        pixel_values, image_grid_thw = deserialize_mm_features(latest_features)

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
        )

    async def generate_batched(self, *args, **kwargs) -> GeneratorOutput:
        raise NotImplementedError(
            "SkyRLVLMGymGenerator does not support batched generation. "
            "Use the default async agent_loop path instead."
        )
