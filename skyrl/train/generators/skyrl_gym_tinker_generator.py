"""
SkyRLGymTinkerGenerator: multi-modal multi-turn RL generator using Tinker chunks.

Uses a Renderer to convert messages → ModelInput(chunks) and
RemoteInferenceClient.sample() for generation. Drop-in replacement for
SkyRLGymGenerator with the same GeneratorInput/GeneratorOutput interface.
"""

import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from tqdm.asyncio import tqdm

import skyrl_gym
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.tinker import types
from skyrl.tinker.types import EncodedTextChunk, ModelInput
from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.utils import (
    get_rollout_metrics,
)
from skyrl.train.renderers.base import (
    Message,
    Renderer,
    get_text_content,
)
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput


@dataclass
class TokensWithLogprobs:
    tokens: list[int]
    logprobs: list[float]
    stop_reason: str


@dataclass
class Transition:
    observation: ModelInput
    action: TokensWithLogprobs


@dataclass
class AgentLoopState:
    history: List[Message]
    transitions: List[Transition]


def _is_prefix(seq1: list[types.ModelInputChunk | int], seq2: list[types.ModelInputChunk | int]) -> bool:
    """
    Check if seq1 is a prefix of seq2.
    """
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def _extract_single_agent_loop_state(
    agent_loop_state: AgentLoopState,
) -> GeneratorOutput:
    """
    Returns fields
    """
    return {
        "response_ids": None,
        "reward": None,
        "stop_reason": None,
        "loss_mask": None,
        "prompt_ids": None,
        "env_metrics": None,
        "model_input": None,
    }


def agent_loop_states_to_generator_output(
    agent_loop_states: List[AgentLoopState],
) -> GeneratorOutput:
    """
    Converts the agent loop state to a generator output.
    """
    outputs = [_extract_single_agent_loop_state(state) for state in agent_loop_states]
    return GeneratorOutput(
        prompt_token_ids=[output["prompt_ids"] for output in outputs],
        response_ids=[output["response_ids"] for output in outputs],
        rewards=[output["reward"] for output in outputs],
        loss_masks=[output["loss_mask"] for output in outputs],
        stop_reasons=[output["stop_reason"] for output in outputs],
        rollout_metrics=None,
        rollout_logprobs=None,
    )


def get_rollout_metrics(
    responses: List[List[int]],
    rewards: Union[List[float], List[List[float]]],
    env_metrics: Optional[List[Dict[str, Any]]] = None,
    env_classes: Optional[List[str]] = None,
):
    """
    Computes rollout metrics including token statistics and optional environment-specific metrics.

    Args:
        responses: List of token ID sequences for each response
        rewards: List of rewards (either per-trajectory or per-token)
        env_metrics: Optional list of environment-specific metrics for each trajectory
        env_classes: Optional list of environment class names for each trajectory

    Returns:
        Dictionary of aggregated metrics
    """
    num_tokens_arr = np.array([len(response) for response in responses])
    # Support both response-level and token-level rewards
    flat_rewards = []
    for r in rewards:
        if isinstance(r, list):
            flat_rewards.append(float(np.sum(r)))
        else:
            flat_rewards.append(float(r))
    flat_rewards_arr = np.array(flat_rewards)
    non_zero_rewards_arr = flat_rewards_arr > 0.0
    zero_rewards_arr = flat_rewards_arr == 0.0
    # average tokens for non zero rewards
    avg_tokens_non_zero_rewards = (
        np.mean(num_tokens_arr[non_zero_rewards_arr]) if non_zero_rewards_arr.sum() > 0 else np.zeros(1)
    )
    # average tokens for zero rewards
    avg_tokens_zero_rewards = np.mean(num_tokens_arr[zero_rewards_arr]) if zero_rewards_arr.sum() > 0 else np.zeros(1)

    rollout_metrics = {
        "generate/min_num_tokens": np.min(num_tokens_arr).item(),
        "generate/max_num_tokens": np.max(num_tokens_arr).item(),
        "generate/avg_num_tokens": np.mean(num_tokens_arr).item(),
        "generate/std_num_tokens": np.std(num_tokens_arr).item(),
        "generate/avg_tokens_non_zero_rewards": avg_tokens_non_zero_rewards.item(),
        "generate/avg_tokens_zero_rewards": avg_tokens_zero_rewards.item(),
    }

    if env_metrics is not None and env_classes is not None:
        env_to_metrics = defaultdict(list)
        for i, metrics in enumerate(env_metrics):
            env_to_metrics[env_classes[i]].append(metrics)
        for env_name, metrics in env_to_metrics.items():
            # Aggregate metrics across all trajectories for the same environment
            agg = aggregate_for_environment(env_name, metrics)
            for key, value in agg.items():
                rollout_metrics[f"environment/{key}"] = value

    return rollout_metrics


### util methods


def _model_input_length(model_input: ModelInput) -> int:
    """
    Returns the total length of the model input in tokens.
    """
    total_length = 0
    for chunk in model_input.chunks:
        if isinstance(chunk, EncodedTextChunk):
            total_length += len(chunk.tokens)
        elif hasattr(chunk, "expected_tokens"):
            total_length += chunk.expected_tokens
    return total_length


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
                max_workers=self.skyrl_gym_cfg.max_env_workers,
                thread_name_prefix="skyrl-gym-env-",
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
    ) -> AgentLoopState:
        """Append-only multi-turn generation loop using Tinker chunks.

        Builds the initial prompt once, then appends response tokens and
        observation chunks to a running list — no re-rendering required.

        Flow per turn:
        1. ModelInput(running_chunks + suffix) → client.sample() → response tokens
        2. renderer.parse_response(tokens) → text → env.step() → observations
        """
        # Init environment
        env_extras = dict(env_extras)
        env_extras["max_turns"] = self.max_turns
        env_config = getattr(self.skyrl_gym_cfg, env_class, dict())
        env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extras)

        session_id = trajectory_id.to_string() if trajectory_id is not None else uuid4().hex

        # Init conversation
        messages_raw, _ = await self._run_in_executor_if_available(env.init, copy.deepcopy(prompt))
        initial_messages: List[Message] = self._convert_conversation(messages_raw)
        agent_loop_state = AgentLoopState(history=initial_messages)

        stop_sequences = self.renderer.get_stop_sequences()
        # Build sampling params for the sample() call
        if stop_sequences:
            sample_params["stop_token_ids"] = stop_sequences
        sample_params = self._build_sample_params(sampling_params, max_tokens)

        get_logprobs = self.generator_cfg.sampling_params.logprobs is not None

        # State tracking
        rollout_logprobs: Optional[List[float]] = [] if get_logprobs else None  # TODO (nithinc): impl
        stop_reason = "length"
        done = False

        while not done:
            model_input = self.renderer.build_generation_prompt(agent_loop_state.history)
            model_input_length = _model_input_length(model_input)
            if model_input_length > max_input_length:
                break

            # Sample from the model
            sample_response = await self.client.sample(
                {
                    "json": {
                        "prompt": model_input.model_dump(),
                        "num_samples": 1,
                        "sampling_params": sample_params,
                        "session_id": session_id,
                    }
                }
            )

            seq = sample_response["sequences"][0]
            response_tokens: List[int] = seq["tokens"]
            response_logprobs: Optional[List[float]] = seq.get("logprobs")  # TODO also need to add prompt logprobs
            stop_reason = seq.get("stop_reason", "length")

            action = TokensWithLogprobs(
                tokens=response_tokens,
                logprobs=response_logprobs,
                stop_reason=stop_reason,
            )
            transition = Transition(observation=model_input, action=action)
            agent_loop_state.transitions.append(transition)

            # Parse response tokens → Message
            assistant_message, _parse_ok = self.renderer.parse_response(response_tokens)
            assert _parse_ok, "Failed to parse response"
            agent_loop_state.history.append(assistant_message)
            response_text = get_text_content(assistant_message)

            # Environment step
            env_step_output: BaseTextEnvStepOutput = await self._run_in_executor_if_available(env.step, response_text)
            step_reward: float = env_step_output["reward"]
            done = env_step_output["done"]
            new_obs = env_step_output["observations"]
            new_obs_messages: List[Message] = self._convert_conversation(new_obs)
            agent_loop_state.history.extend(new_obs_messages)

        # Get final metrics and close env
        env_metrics = env.get_metrics()
        await self._run_in_executor_if_available(env.close)
        return agent_loop_state

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

        all_agent_loop_states: List[AgentLoopState] = await tqdm.gather(
            *tasks,
            desc="Generating Trajectories (Tinker)",
            miniters=max(1, len(tasks) // 10),
            mininterval=5,
            disable=disable_tqdm,
        )

        return agent_loop_states_to_generator_output(all_agent_loop_states)

        responses = [output.response_ids for output in all_outputs]
        rewards = [output.reward for output in all_outputs]
        stop_reasons = [output.stop_reason for output in all_outputs]
        loss_masks = [output.loss_mask for output in all_outputs]
        prompt_token_ids = [output.prompt_ids for output in all_outputs]
        env_metrics = [output.env_metrics for output in all_outputs]
        model_inputs = [output.model_input for output in all_outputs]

        get_logprobs = self.generator_cfg.sampling_params.logprobs is not None
        if get_logprobs:
            rollout_logprobs = [output.rollout_logprobs for output in all_outputs]
        else:
            rollout_logprobs = None

        rollout_metrics = get_rollout_metrics(responses, rewards, env_metrics, env_classes)

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

    def _build_sample_params(self, override_params: Optional[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
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
