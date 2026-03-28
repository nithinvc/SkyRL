"""
SkyRLGymTinkerGenerator: multi-modal multi-turn RL generator using Tinker chunks.

Uses a Renderer to convert messages → ModelInput(chunks) and
RemoteInferenceClient.sample() for generation. Drop-in replacement for
SkyRLGymGenerator with the same GeneratorInput/GeneratorOutput interface.
"""

from __future__ import annotations

import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from loguru import logger
from tqdm.asyncio import tqdm

import skyrl_gym
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient
from skyrl.tinker.types import EncodedTextChunk, ImageChunk, ModelInput
from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.utils import (
    apply_overlong_filtering,
    get_rollout_metrics,
)
from skyrl.train.renderers.base import Message, RenderContext, Renderer, get_text_content
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput


@dataclass
class TrajectoryOutput:
    """Output from a single agent_loop execution."""

    response_ids: List[int]
    reward: Union[List[float], float]
    stop_reason: str
    loss_mask: List[int]
    prompt_ids: List[int]
    rollout_logprobs: Optional[List[float]]
    env_metrics: Dict[str, Any]
    model_input: Optional[ModelInput] = None


def _count_chunk_tokens(chunks) -> int:
    """Count all tokens in a list of chunks (text tokens + image placeholder tokens)."""
    total = 0
    for c in chunks:
        if isinstance(c, EncodedTextChunk):
            total += len(c.tokens)
        elif isinstance(c, ImageChunk) and c.expected_tokens is not None:
            total += c.expected_tokens
    return total


def _flatten_text_tokens(model_input: ModelInput) -> List[int]:
    """Extract all text token IDs from a ModelInput."""
    tokens: List[int] = []
    for chunk in model_input.chunks:
        if isinstance(chunk, EncodedTextChunk):
            tokens.extend(chunk.tokens)
    return tokens


class SkyRLGymTinkerGenerator(GeneratorInterface):
    """Multi-modal multi-turn generator using Tinker chunks + RemoteInferenceClient.sample().

    Uses a Renderer to handle chat template formatting and image encoding.
    Accumulates chunks append-only during the generation loop to avoid
    tokenization boundary mismatches from re-rendering.
    """

    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        skyrl_gym_cfg: SkyRLGymConfig,
        client: RemoteInferenceClient,
        renderer: Renderer,
    ):
        self.generator_cfg = generator_cfg
        self.skyrl_gym_cfg = skyrl_gym_cfg
        self.client = client
        self.renderer = renderer
        self.max_turns = generator_cfg.max_turns

        if self.skyrl_gym_cfg.max_env_workers > 0:
            self.env_executor = ThreadPoolExecutor(
                max_workers=self.skyrl_gym_cfg.max_env_workers, thread_name_prefix="skyrl-gym-env-"
            )
        else:
            self.env_executor = None

    async def _run_in_executor_if_available(self, func, *args, **kwargs):
        if (executor := self.env_executor) is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    # -- agent loop -----------------------------------------------------------

    async def agent_loop(
        self,
        prompt: List[Dict[str, Any]],
        env_class: str,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        trajectory_id: Optional[TrajectoryID] = None,
    ) -> TrajectoryOutput:
        """Append-only multi-turn generation loop using Tinker chunks.

        Builds the initial prompt once, then appends response tokens and
        observation chunks to a running list — no re-rendering required.

        Flow per turn:
        1. ModelInput(running_chunks + suffix) → client.sample() → response tokens
        2. Append suffix + response + <|im_end|> to running_chunks
        3. renderer.parse_response(tokens) → text → env.step() → observations
        4. renderer.render_message(obs) → chunks → append to running_chunks
        """
        # Init environment
        env_extras = dict(env_extras)
        env_extras["max_turns"] = self.max_turns
        env_config = getattr(self.skyrl_gym_cfg, env_class, dict())
        env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extras)

        session_id = trajectory_id.to_string() if trajectory_id is not None else uuid4().hex

        # Init conversation
        messages_raw, _ = await self._run_in_executor_if_available(env.init, copy.deepcopy(prompt))
        messages: List[Message] = self._convert_conversation(messages_raw)

        # Build initial prompt ONCE and split off the generation suffix.
        # The suffix (e.g. \n<|im_start|>assistant\n) is always the last chunk
        # because Qwen3 has no BOS tokens and we don't use prefill.
        model_input = self.renderer.build_generation_prompt(messages)
        initial_prompt_tokens = _flatten_text_tokens(model_input)

        all_initial_chunks = list(model_input.chunks)
        suffix_chunk = all_initial_chunks[-1]
        assert isinstance(suffix_chunk, EncodedTextChunk), (
            f"Expected suffix to be EncodedTextChunk, got {type(suffix_chunk)}"
        )
        suffix_tokens = list(suffix_chunk.tokens)

        # running_chunks holds the conversation WITHOUT the suffix.
        # Before each sample() call we append the suffix temporarily.
        running_chunks: List = list(all_initial_chunks[:-1])
        running_token_count = _count_chunk_tokens(running_chunks)

        # Stop token: vLLM excludes this from response token_ids, so we
        # must append it explicitly to running_chunks after each response.
        stop_sequences = self.renderer.get_stop_sequences()
        end_message_token: int = stop_sequences[0]

        # Build sampling params for the sample() call
        sample_params = self._build_sample_params(sampling_params, max_tokens)
        if stop_sequences:
            sample_params["stop_token_ids"] = stop_sequences

        get_logprobs = self.generator_cfg.sampling_params.logprobs is not None

        # State tracking
        all_response_ids: List[int] = []
        loss_mask: List[int] = []
        rollout_logprobs: Optional[List[float]] = [] if get_logprobs else None
        per_step_rewards: List[Tuple[float, Optional[int]]] = []
        stop_reason = "length"
        done = False

        while not done:
            # Length check: running tokens + suffix tokens
            if running_token_count + len(suffix_tokens) > max_input_length:
                break

            # Build model input = running_chunks + suffix
            model_input = ModelInput(chunks=running_chunks + [suffix_chunk])

            # Sample from the model
            sample_response = await self.client.sample({
                "json": {
                    "prompt": model_input.model_dump(),
                    "num_samples": 1,
                    "sampling_params": sample_params,
                    "session_id": session_id,
                }
            })

            seq = sample_response["sequences"][0]
            response_tokens: List[int] = seq["tokens"]
            response_logprobs: Optional[List[float]] = seq.get("logprobs")
            stop_reason = seq.get("stop_reason", "length")

            # Parse response tokens → Message
            assistant_message, _parse_ok = self.renderer.parse_response(response_tokens)
            response_text = get_text_content(assistant_message)

            # Environment step
            env_step_output: BaseTextEnvStepOutput = await self._run_in_executor_if_available(env.step, response_text)
            step_reward: float = env_step_output["reward"]
            done = env_step_output["done"]
            new_obs = env_step_output["observations"]

            # Track response tokens in all_response_ids (loss_mask=1)
            response_end_idx = len(all_response_ids) + len(response_tokens) - 1
            all_response_ids.extend(response_tokens)
            loss_mask.extend([1] * len(response_tokens))
            if rollout_logprobs is not None:
                if response_logprobs:
                    rollout_logprobs.extend(response_logprobs)
                else:
                    rollout_logprobs.extend([0.0] * len(response_tokens))

            # Append to running_chunks:
            # - suffix becomes the assistant header in conversation history
            # - response tokens (the model's actual output)
            # - <|im_end|> token (excluded by vLLM from response, must add explicitly)
            running_chunks.append(EncodedTextChunk(tokens=suffix_tokens))
            running_chunks.append(EncodedTextChunk(tokens=response_tokens))
            running_chunks.append(EncodedTextChunk(tokens=[end_message_token]))
            running_token_count += len(suffix_tokens) + len(response_tokens) + 1

            # Keep messages list updated (needed for final training model_input)
            messages.append(assistant_message)
            obs_messages = self._convert_conversation(new_obs)
            messages.extend(obs_messages)

            # Render and append observation chunks
            if not done and obs_messages:
                obs_token_count = 0
                for obs_msg in obs_messages:
                    ctx = RenderContext(idx=1, is_last=False)
                    rendered_obs = self.renderer.render_message(obs_msg, ctx)
                    if rendered_obs.header:
                        running_chunks.append(rendered_obs.header)
                        obs_token_count += len(rendered_obs.header.tokens)
                    for chunk in rendered_obs.output:
                        if chunk:
                            running_chunks.append(chunk)
                            if isinstance(chunk, EncodedTextChunk):
                                obs_token_count += len(chunk.tokens)
                            elif isinstance(chunk, ImageChunk) and chunk.expected_tokens is not None:
                                obs_token_count += chunk.expected_tokens

                running_token_count += obs_token_count

                # Observation token accounting for all_response_ids:
                # Includes suffix (assistant header) + end_token + observation tokens.
                # The suffix is counted because it sits between the response tokens and
                # the observations in the actual sequence (it becomes the assistant header
                # in conversation history). This matches the re-render approach where
                # obs_count = new_total - old_total - response_len.
                total_obs_token_count = len(suffix_tokens) + 1 + obs_token_count
                all_response_ids.extend([0] * total_obs_token_count)
                loss_mask.extend([0] * total_obs_token_count)
                if rollout_logprobs is not None:
                    rollout_logprobs.extend([0.0] * total_obs_token_count)

            per_step_rewards.append((step_reward, response_end_idx))

        # Get final metrics and close env
        env_metrics = env.get_metrics()
        await self._run_in_executor_if_available(env.close)

        # Trim trailing observation tokens from response_ids / loss_mask
        # (same as existing generator: remove tokens after last response_end_idx)
        if per_step_rewards:
            last_response_end = per_step_rewards[-1][1]
            if last_response_end is not None and last_response_end + 1 < len(all_response_ids):
                all_response_ids = all_response_ids[: last_response_end + 1]
                loss_mask = loss_mask[: last_response_end + 1]
                if rollout_logprobs is not None:
                    rollout_logprobs = rollout_logprobs[: last_response_end + 1]

        # Build per-token rewards
        reward_out = self._build_per_token_rewards(per_step_rewards, all_response_ids)

        # Final model input (with images) for training rendering
        final_model_input = self.renderer.build_generation_prompt(messages)

        return TrajectoryOutput(
            response_ids=all_response_ids,
            reward=reward_out,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=initial_prompt_tokens,
            rollout_logprobs=rollout_logprobs,
            env_metrics=env_metrics,
            model_input=final_model_input,
        )

    # -- generate (batch orchestration) ----------------------------------------

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        """Generate trajectories for the input batch.

        Launches parallel agent_loop tasks and aggregates into GeneratorOutput.
        """
        prompts = input_batch["prompts"]
        env_classes = input_batch["env_classes"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch.get("trajectory_ids", None)
        sampling_params: Optional[dict] = input_batch.get("sampling_params", None)
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length

        tasks = []
        for i in range(len(prompts)):
            tasks.append(
                self.agent_loop(
                    prompts[i],
                    env_classes[i],
                    env_extras[i],
                    max_tokens,
                    max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[i] if trajectory_ids is not None else None,
                )
            )

        all_outputs: List[TrajectoryOutput] = await tqdm.gather(
            *tasks,
            desc="Generating Trajectories (Tinker)",
            miniters=max(1, len(tasks) // 10),
            mininterval=5,
            disable=disable_tqdm,
        )

        responses = [output.response_ids for output in all_outputs]
        rewards = [output.reward for output in all_outputs]
        stop_reasons = [output.stop_reason for output in all_outputs]
        loss_masks = [output.loss_mask for output in all_outputs]
        prompt_token_ids = [output.prompt_ids for output in all_outputs]
        env_metrics = [output.env_metrics for output in all_outputs]

        get_logprobs = self.generator_cfg.sampling_params.logprobs is not None
        if get_logprobs:
            rollout_logprobs = [output.rollout_logprobs for output in all_outputs]
        else:
            rollout_logprobs = None

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
            "rollout_expert_indices": None,
            "is_last_step": None,
        }

        return generator_output

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _convert_conversation(messages_raw: List[Dict[str, Any]]) -> List[Message]:
        """Convert ConversationType dicts to renderer Message objects.

        Normalizes OpenAI-format image_url parts to renderer ImagePart format.
        VisGym returns {"type": "image_url", "image_url": {"url": "data:..."}}
        but the renderer expects {"type": "image", "image": <str>}.
        """
        result: List[Message] = []
        for m in messages_raw:
            content = m["content"]
            if isinstance(content, list):
                normalized_parts = []
                for part in content:
                    if part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        normalized_parts.append({"type": "image", "image": url})
                    else:
                        normalized_parts.append(part)
                content = normalized_parts
            result.append(Message(role=m["role"], content=content))
        return result

    def _build_sample_params(
        self, override_params: Optional[Dict[str, Any]], max_tokens: int
    ) -> Dict[str, Any]:
        """Build Tinker-format sampling params from config + overrides."""
        cfg = self.generator_cfg.sampling_params
        params: Dict[str, Any] = {
            "temperature": cfg.temperature,
            "max_tokens": max_tokens,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
        }
        if cfg.stop:
            params["stop"] = cfg.stop

        # Apply overrides (already in Tinker format from GeneratorInput)
        if override_params:
            params.update(override_params)

        return params

    def _build_per_token_rewards(
        self,
        per_step_rewards: List[Tuple[float, Optional[int]]],
        response_ids: List[int],
    ) -> Union[float, List[float]]:
        """Build per-token rewards placed at assistant turn boundaries."""
        token_level_rewards: List[float] = [0.0] * len(response_ids)
        for step_reward, idx in per_step_rewards:
            if idx is not None and idx < len(response_ids):
                token_level_rewards[idx] += step_reward
        return token_level_rewards

    @staticmethod
    def _zero_reward_if_not_stop(
        rewards: List[Union[float, List[float]]], stop_reasons: List[str]
    ) -> List[Union[float, List[float]]]:
        """Zero out rewards for trajectories that didn't stop normally."""
        result = []
        for reward, stop_reason in zip(rewards, stop_reasons):
            if stop_reason != "stop":
                if isinstance(reward, list):
                    result.append([0.0] * len(reward))
                else:
                    result.append(0.0)
            else:
                result.append(reward)
        return result
