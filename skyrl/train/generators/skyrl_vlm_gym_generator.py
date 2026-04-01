"""
SkyRLVLMGymGenerator: VLM (vision-language model) multi-turn RL generator.

Subclasses SkyRLGymGenerator to handle multi-modal observations (images)
from VisGym environments. Uses pure Python types (OpenAI-format messages
with base64-encoded images) rather than Tinker chunks.
"""

from typing import Any, Dict, Optional

from loguru import logger

from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
    InferenceEngineClient,
)
from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.base import GeneratorOutput, TrajectoryID
from skyrl.train.generators.skyrl_gym_generator import (
    SkyRLGymGenerator,
    TrajectoryOutput,
)


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
        inference_engine_client: InferenceEngineClient,
        tokenizer,
    ):
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer)
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

        Handles multi-modal observations (OpenAI-format messages with
        embedded base64 images) from VisGym environments.

        Args:
            prompt: Initial conversation messages (may contain multi-modal content)
            env_class: Environment class name (e.g., "visgym")
            env_extras: Extra configuration passed to the environment
            max_tokens: Maximum number of tokens to generate per turn
            max_input_length: Maximum total input length before truncation
            sampling_params: Optional override sampling parameters
            trajectory_id: Optional trajectory identifier for session tracking

        Returns:
            TrajectoryOutput with response_ids, reward, stop_reason,
            loss_mask, prompt_ids, and rollout_logprobs.
        """
        # TODO: Implement VLM agent loop
        raise NotImplementedError("SkyRLVLMGymGenerator.agent_loop() is not yet implemented.")

    async def generate_batched(self, *args, **kwargs) -> GeneratorOutput:
        raise NotImplementedError(
            "SkyRLVLMGymGenerator does not support batched generation. "
            "Use the default async agent_loop path instead."
        )
