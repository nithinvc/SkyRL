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
from skyrl.train.renderers.base import Message, Renderer, get_text_content
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


def _count_text_tokens(model_input: ModelInput) -> int:
    """Count text tokens in a ModelInput (excludes image placeholder tokens)."""
    return sum(len(c.tokens) for c in model_input.chunks if isinstance(c, EncodedTextChunk))


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
    Re-renders the full conversation each turn (extension property ensures prefix consistency).
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
        """Multi-turn generation loop using Tinker chunks.

        Flow per turn:
        1. renderer.build_generation_prompt(messages) → ModelInput(chunks)
        2. client.sample(chunks) → response tokens + logprobs
        3. renderer.parse_response(tokens) → assistant Message
        4. env.step(text) → observations + reward
        5. Append assistant + observations to messages, repeat
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

        # Build initial prompt and measure its length
        model_input = self.renderer.build_generation_prompt(messages)
        initial_prompt_tokens = _flatten_text_tokens(model_input)
        prompt_token_count = len(initial_prompt_tokens)

        # Build sampling params for the sample() call
        sample_params = self._build_sample_params(sampling_params, max_tokens)
        stop_sequences = self.renderer.get_stop_sequences()
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
            # Re-render the full conversation each turn
            model_input = self.renderer.build_generation_prompt(messages)
            current_token_count = _count_text_tokens(model_input)

            if current_token_count > max_input_length:
                break

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

            # Track response tokens
            response_end_idx = len(all_response_ids) + len(response_tokens) - 1
            all_response_ids.extend(response_tokens)
            loss_mask.extend([1] * len(response_tokens))
            if rollout_logprobs is not None:
                if response_logprobs:
                    rollout_logprobs.extend(response_logprobs)
                else:
                    rollout_logprobs.extend([0.0] * len(response_tokens))

            # Update messages with assistant response + observations
            messages.append(assistant_message)
            obs_messages = self._convert_conversation(new_obs)
            messages.extend(obs_messages)

            # Compute observation token count by re-rendering
            if not done and obs_messages:
                new_model_input = self.renderer.build_generation_prompt(messages)
                new_token_count = _count_text_tokens(new_model_input)
                # obs tokens = new total - old total - response tokens
                # (old total includes the generation suffix, new total also includes it,
                #  so the suffix cancels out)
                obs_token_count = new_token_count - current_token_count - len(response_tokens)

                if obs_token_count > 0:
                    # Observation tokens get loss_mask=0 and dummy logprobs
                    all_response_ids.extend([0] * obs_token_count)
                    loss_mask.extend([0] * obs_token_count)
                    if rollout_logprobs is not None:
                        rollout_logprobs.extend([0.0] * obs_token_count)

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

        ConversationType messages are already compatible with the Message TypedDict —
        they have 'role' and 'content' keys. Content may be a string or a list of
        content parts (text, image, etc).
        """
        return [Message(role=m["role"], content=m["content"]) for m in messages_raw]

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
