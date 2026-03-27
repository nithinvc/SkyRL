from typing import Any, Dict, List, Tuple

import gymnasium

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.visgym.utils import extract_action, make_image_message


class VisGymEnv(BaseTextEnv):
    """Wraps a VisGym environment as a BaseTextEnv for use with SkyRLGymGenerator.

    Bridges VisGym's gymnasium.Env interface (image observations, tuple-string actions,
    binary rewards) to SkyRL-Gym's BaseTextEnv interface (OpenAI message format, raw text
    actions, float rewards).

    Configuration via extras dict:
        visgym_env_id (str): VisGym environment ID, e.g. "maze_2d/easy"
        seed (int, optional): Random seed for environment reset
        max_turns (int, optional): Maximum steps per episode (default: 10)
        visgym_kwargs (dict, optional): Extra kwargs passed to gymnasium.make()
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "visgym_env_id" in extras, "visgym_env_id is required in extras"
        self.visgym_env_id = extras["visgym_env_id"]
        self.seed_value = extras.get("seed", None)
        self.max_turns = extras.get("max_turns", 10)
        visgym_kwargs = extras.get("visgym_kwargs", {})

        self.visgym_env = gymnasium.make(self.visgym_env_id, **visgym_kwargs)

        # Tracking
        self.step_count = 0
        self.parse_failures = 0

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Reset the VisGym env and return the initial multimodal prompt."""
        obs, info = self.visgym_env.reset(seed=self.seed_value)

        task_prompt = self.visgym_env.get_prompt()
        image = self.visgym_env.render()

        user_msg = make_image_message(task_prompt, image)
        initial_prompt = [user_msg]

        return initial_prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.step_count += 1

        # Extract action tuple from VLM output (strips reasoning text)
        extracted, matched = extract_action(action)
        if not matched:
            self.parse_failures += 1

        # Step the VisGym environment
        obs, reward, terminated, truncated, info = self.visgym_env.step(extracted)

        done = terminated or truncated or self.step_count >= self.max_turns

        if not done:
            # Build multimodal observation with updated image + feedback
            image = self.visgym_env.render()
            feedback = info.get("env_feedback", None) or ""
            if not feedback:
                feedback = "Action executed. Here is the current state:"
            obs_msg = make_image_message(feedback, image)
            observations = [obs_msg]
        else:
            observations = []

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=float(reward),
            done=done,
            metadata={
                "env_feedback": info.get("env_feedback", ""),
                "terminated": terminated,
                "truncated": truncated,
                "step_count": self.step_count,
                "extracted_action": extracted,
            },
        )

    def close(self):
        self.visgym_env.close()

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "step_count": self.step_count,
            "parse_failures": self.parse_failures,
            "visgym_env_id": self.visgym_env_id,
        }

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics:
            return {}
        numeric_keys = ["step_count", "parse_failures"]
        result = {}
        for key in numeric_keys:
            values = [m[key] for m in metrics if key in m]
            if values:
                result[f"avg_{key}"] = sum(values) / len(values)
        return result
