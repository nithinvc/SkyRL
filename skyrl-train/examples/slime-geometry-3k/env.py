# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Environment for Geometry-3K multi-modal math problems.
"""

import re
from typing import Any, Dict

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    answer = answer.strip().lower()
    # Remove common punctuation and extra whitespace
    answer = re.sub(r"[.,;:!?]$", "", answer)
    answer = re.sub(r"\s+", " ", answer)
    return answer


def extract_answer(completion: str) -> str:
    """Extract the answer from model completion using <answer> tags."""
    # Try to extract from <answer> tags first
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: try to find the last line or sentence that looks like an answer
    # This handles cases where the model doesn't use the expected format
    lines = completion.strip().split("\n")
    if lines:
        return lines[-1].strip()

    return completion.strip()


def compute_score(completion: str, ground_truth: str) -> float:
    """
    Compute the reward score for a geometry problem.

    Args:
        completion: The model's completion/response
        ground_truth: The ground truth answer

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    try:
        # Extract answer from completion
        student_answer = extract_answer(completion)

        # Normalize both answers
        student_normalized = normalize_answer(student_answer)
        ground_truth_normalized = normalize_answer(ground_truth)

        # Exact match after normalization
        if student_normalized == ground_truth_normalized:
            return 1.0

        # Try numeric comparison for numerical answers
        try:
            student_num = float(re.sub(r"[^\d.\-]", "", student_answer))
            ground_truth_num = float(re.sub(r"[^\d.\-]", "", ground_truth))
            if abs(student_num - ground_truth_num) < 1e-6:
                return 1.0
        except (ValueError, TypeError):
            pass

        # Check if student answer contains the ground truth (for multiple choice)
        # e.g., "The answer is A" should match ground truth "A"
        if ground_truth_normalized in student_normalized:
            return 1.0

        return 0.0

    except Exception:
        return 0.0


class Geometry3kEnv(BaseTextEnv):
    """
    Environment for Geometry-3K multi-modal math problems.

    This environment evaluates model responses against ground truth answers
    for geometry problems that include diagrams/images.
    """

    def __init__(
        self,
        env_config: Dict[str, Any] = {},
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

        self.max_turns = extras.get("max_turns", 1)

    def _compute_reward(self, action: str) -> float:
        """Compute the reward for the given action (model response)."""
        return compute_score(action, self.ground_truth)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Execute one step in the environment.

        For geometry problems, we evaluate the model's response against
        the ground truth answer. This is a single-turn environment.

        Args:
            action: The model's response/completion

        Returns:
            BaseTextEnvStepOutput with reward and done=True
        """
        done = True  # Single-turn environment
        reward = self._compute_reward(action)
        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=done,
            metadata={"ground_truth": self.ground_truth},
        )