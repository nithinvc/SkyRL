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

Uses boxed-answer extraction and sympy-based math grading ported from slime.
"""

import importlib
from typing import Any, Dict

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

# The directory name "geometry-3k" contains a hyphen, which isn't valid in
# standard Python import syntax.  Use importlib so the module path matches
# the on-disk layout used by the entry-point registration.
_math_utils = importlib.import_module("examples.geometry-3k.math_utils")
grade_answer_verl = _math_utils.grade_answer_verl


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
        try:
            if grade_answer_verl(action, self.ground_truth):
                return 1.0
            return 0.0
        except Exception:
            return 0.0

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
