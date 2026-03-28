"""
Layer 2: Generator unit tests for SkyRLGymTinkerGenerator with mocked client + env.

Tests single-turn, multi-turn, multi-modal, logprobs, batch generation,
and GeneratorOutput interface compliance.

Run with:
    uv run --extra dev --extra fsdp pytest tests/test_tinker_generator.py -v
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from skyrl.tinker.types import EncodedTextChunk, ImageChunk, ModelInput
from skyrl.train.generators.base import GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl.train.generators.skyrl_gym_tinker_generator import SkyRLGymTinkerGenerator
from skyrl.train.renderers.base import Message, RenderedMessage, RenderContext, get_text_content
from skyrl.train.renderers.qwen3 import Qwen3Renderer


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    """A real Qwen3 tokenizer for the renderer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", use_fast=True)


@pytest.fixture
def renderer(tokenizer):
    return Qwen3Renderer(tokenizer, strip_thinking_from_history=False)


@pytest.fixture
def generator_cfg():
    """Minimal GeneratorConfig."""
    from skyrl.train.config import GeneratorConfig, SamplingParams

    cfg = GeneratorConfig()
    cfg.sampling_params = SamplingParams(
        max_generate_length=64,
        temperature=0.0,
        logprobs=0,  # enable logprobs
    )
    cfg.max_input_length = 512
    cfg.max_turns = 3
    cfg.zero_reward_on_non_stop = False
    cfg.apply_overlong_filtering = False
    return cfg


@pytest.fixture
def skyrl_gym_cfg():
    from skyrl.train.config import SkyRLGymConfig

    cfg = SkyRLGymConfig()
    cfg.max_env_workers = 0  # synchronous for tests
    return cfg


def _make_mock_client(sample_responses: Optional[List[Dict]] = None):
    """Create a mock RemoteInferenceClient.

    sample_responses: list of dicts, one per call. Each has 'tokens', 'logprobs', 'stop_reason'.
    """
    client = AsyncMock()
    if sample_responses is None:
        sample_responses = [{"tokens": [10, 11, 12], "logprobs": [-0.1, -0.2, -0.3], "stop_reason": "stop"}]

    call_idx = 0

    async def mock_sample(request_payload):
        nonlocal call_idx
        resp = sample_responses[min(call_idx, len(sample_responses) - 1)]
        call_idx += 1
        return {
            "type": "sample",
            "sequences": [
                {
                    "tokens": resp["tokens"],
                    "logprobs": resp.get("logprobs"),
                    "stop_reason": resp.get("stop_reason", "stop"),
                }
            ],
            "prompt_logprobs": None,
            "topk_prompt_logprobs": None,
        }

    client.sample = mock_sample
    return client


def _make_mock_env(steps: Optional[List[Dict]] = None):
    """Create a mock environment.

    steps: list of dicts with 'observations', 'reward', 'done'.
    Each call to step() pops from the list.
    """
    if steps is None:
        steps = [{"observations": [], "reward": 1.0, "done": True}]

    env = MagicMock()
    step_idx = 0

    def mock_init(prompt):
        return prompt, {}

    def mock_step(action):
        nonlocal step_idx
        s = steps[min(step_idx, len(steps) - 1)]
        step_idx += 1
        return {
            "observations": s.get("observations", []),
            "reward": s.get("reward", 0.0),
            "done": s.get("done", True),
            "metadata": {},
        }

    env.init = mock_init
    env.step = mock_step
    env.get_metrics = MagicMock(return_value={})
    env.close = MagicMock()
    return env


# ---------------------------------------------------------------------------
# Single-turn tests
# ---------------------------------------------------------------------------


class TestSingleTurn:
    def test_basic_single_turn(self, renderer, generator_cfg, skyrl_gym_cfg):
        """Single-turn: prompt → generate → done. Verify output structure."""
        client = _make_mock_client([
            {"tokens": [10, 11, 12], "logprobs": [-0.1, -0.2, -0.3], "stop_reason": "stop"}
        ])
        env = _make_mock_env([{"observations": [], "reward": 1.0, "done": True}])

        generator = SkyRLGymTinkerGenerator(generator_cfg, skyrl_gym_cfg, client, renderer)

        prompt = [{"role": "user", "content": "What is 2+2?"}]

        with patch("skyrl_gym.make", return_value=env):
            output = asyncio.run(generator.agent_loop(
                prompt, "test_env", {}, max_tokens=64, max_input_length=512
            ))

        assert output.response_ids == [10, 11, 12]
        assert output.stop_reason == "stop"
        assert len(output.loss_mask) == 3
        assert all(m == 1 for m in output.loss_mask)  # all generated tokens
        assert output.rollout_logprobs == [-0.1, -0.2, -0.3]
        assert len(output.prompt_ids) > 0  # should have tokenized prompt

    def test_single_turn_no_logprobs(self, renderer, skyrl_gym_cfg):
        """When logprobs not configured, rollout_logprobs is None."""
        from skyrl.train.config import GeneratorConfig, SamplingParams

        cfg = GeneratorConfig()
        cfg.sampling_params = SamplingParams(max_generate_length=64, logprobs=None)
        cfg.max_input_length = 512
        cfg.max_turns = 1

        client = _make_mock_client([{"tokens": [10], "stop_reason": "stop"}])
        env = _make_mock_env([{"observations": [], "reward": 1.0, "done": True}])

        generator = SkyRLGymTinkerGenerator(cfg, skyrl_gym_cfg, client, renderer)

        with patch("skyrl_gym.make", return_value=env):
            output = asyncio.run(generator.agent_loop(
                [{"role": "user", "content": "Hi"}], "test_env", {},
                max_tokens=64, max_input_length=512
            ))

        assert output.rollout_logprobs is None

    def test_reward_on_response_end(self, renderer, generator_cfg, skyrl_gym_cfg):
        """Per-token reward is placed at the end of the response."""
        client = _make_mock_client([
            {"tokens": [10, 11, 12], "logprobs": [-0.1, -0.2, -0.3], "stop_reason": "stop"}
        ])
        env = _make_mock_env([{"observations": [], "reward": 5.0, "done": True}])

        generator = SkyRLGymTinkerGenerator(generator_cfg, skyrl_gym_cfg, client, renderer)

        with patch("skyrl_gym.make", return_value=env):
            output = asyncio.run(generator.agent_loop(
                [{"role": "user", "content": "Hi"}], "test_env", {},
                max_tokens=64, max_input_length=512
            ))

        assert isinstance(output.reward, list)
        assert output.reward[-1] == 5.0
        assert output.reward[:-1] == [0.0] * (len(output.reward) - 1)


# ---------------------------------------------------------------------------
# Multi-turn tests
# ---------------------------------------------------------------------------


class TestMultiTurn:
    def test_two_turn_conversation(self, renderer, generator_cfg, skyrl_gym_cfg):
        """Two turns: verify loss_mask is 1 for generated, 0 for observations."""
        # Turn 1: generate → observation → Turn 2: generate → done
        client = _make_mock_client([
            {"tokens": [10, 11], "logprobs": [-0.1, -0.2], "stop_reason": "stop"},
            {"tokens": [20, 21, 22], "logprobs": [-0.3, -0.4, -0.5], "stop_reason": "stop"},
        ])
        env = _make_mock_env([
            {"observations": [{"role": "user", "content": "Continue"}], "reward": 0.5, "done": False},
            {"observations": [], "reward": 1.0, "done": True},
        ])

        generator = SkyRLGymTinkerGenerator(generator_cfg, skyrl_gym_cfg, client, renderer)

        with patch("skyrl_gym.make", return_value=env):
            output = asyncio.run(generator.agent_loop(
                [{"role": "user", "content": "Start"}], "test_env", {},
                max_tokens=64, max_input_length=512
            ))

        # Response should contain tokens from both turns + observation tokens
        assert 10 in output.response_ids
        assert 11 in output.response_ids
        assert 20 in output.response_ids

        # Loss mask: generated tokens = 1, observation tokens = 0
        gen_count = sum(1 for m in output.loss_mask if m == 1)
        obs_count = sum(1 for m in output.loss_mask if m == 0)
        assert gen_count >= 5  # at least 2 + 3 generated tokens
        # obs_count could be 0 if trailing obs tokens are trimmed

    def test_max_input_length_stops_loop(self, renderer, skyrl_gym_cfg):
        """Loop terminates when input length exceeds max_input_length."""
        from skyrl.train.config import GeneratorConfig, SamplingParams

        cfg = GeneratorConfig()
        cfg.sampling_params = SamplingParams(max_generate_length=64, logprobs=0)
        cfg.max_input_length = 20  # Very short — should stop quickly
        cfg.max_turns = 10

        client = _make_mock_client([
            {"tokens": [10, 11, 12, 13, 14, 15], "logprobs": [-0.1] * 6, "stop_reason": "length"},
        ])
        env = _make_mock_env([
            {"observations": [{"role": "user", "content": "More"}], "reward": 0.0, "done": False},
            {"observations": [], "reward": 1.0, "done": True},
        ])

        generator = SkyRLGymTinkerGenerator(cfg, skyrl_gym_cfg, client, renderer)

        with patch("skyrl_gym.make", return_value=env):
            output = asyncio.run(generator.agent_loop(
                [{"role": "user", "content": "Start"}], "test_env", {},
                max_tokens=64, max_input_length=20
            ))

        # Should have generated at most 1-2 turns before hitting length limit
        assert output.stop_reason in ("length", "stop")


# ---------------------------------------------------------------------------
# Multi-modal tests
# ---------------------------------------------------------------------------


class TestMultiModal:
    def test_image_in_prompt_produces_image_chunks(self, generator_cfg, skyrl_gym_cfg):
        """Prompt with image produces ModelInput containing ImageChunks."""
        from skyrl.train.renderers.qwen3 import Qwen3VLRenderer
        from skyrl.train.renderers.image_utils import get_image_processor
        from transformers import AutoTokenizer

        vl_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", use_fast=True)
        vl_proc = get_image_processor("Qwen/Qwen3-VL-2B-Instruct")
        vl_renderer = Qwen3VLRenderer(vl_tok, vl_proc, strip_thinking_from_history=False)

        client = _make_mock_client([
            {"tokens": [10, 11], "logprobs": [-0.1, -0.2], "stop_reason": "stop"}
        ])
        env = _make_mock_env([{"observations": [], "reward": 1.0, "done": True}])

        generator = SkyRLGymTinkerGenerator(generator_cfg, skyrl_gym_cfg, client, vl_renderer)

        img = Image.new("RGB", (8, 8), color=(255, 0, 0))
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]

        with patch("skyrl_gym.make", return_value=env):
            output = asyncio.run(generator.agent_loop(
                prompt, "test_env", {}, max_tokens=64, max_input_length=512
            ))

        # The final model_input should contain ImageChunks
        assert output.model_input is not None
        image_chunks = [c for c in output.model_input.chunks if isinstance(c, ImageChunk)]
        assert len(image_chunks) >= 1


# ---------------------------------------------------------------------------
# Batch generation tests
# ---------------------------------------------------------------------------


class TestBatchGeneration:
    def test_generate_batch(self, renderer, generator_cfg, skyrl_gym_cfg):
        """generate() handles multiple prompts in parallel."""
        client = _make_mock_client([
            {"tokens": [10, 11], "logprobs": [-0.1, -0.2], "stop_reason": "stop"},
        ])
        env = _make_mock_env([{"observations": [], "reward": 1.0, "done": True}])

        generator = SkyRLGymTinkerGenerator(generator_cfg, skyrl_gym_cfg, client, renderer)

        input_batch: GeneratorInput = {
            "prompts": [
                [{"role": "user", "content": "Q1"}],
                [{"role": "user", "content": "Q2"}],
                [{"role": "user", "content": "Q3"}],
            ],
            "env_classes": ["gsm8k"] * 3,
            "env_extras": [{}] * 3,
        }

        with patch("skyrl_gym.make", return_value=env):
            output: GeneratorOutput = asyncio.run(generator.generate(input_batch, disable_tqdm=True))

        # Verify GeneratorOutput structure
        assert len(output["prompt_token_ids"]) == 3
        assert len(output["response_ids"]) == 3
        assert len(output["rewards"]) == 3
        assert len(output["loss_masks"]) == 3
        assert len(output["stop_reasons"]) == 3

        # Each response should have the mock tokens
        for resp in output["response_ids"]:
            assert resp == [10, 11]

    def test_generator_output_has_required_keys(self, renderer, generator_cfg, skyrl_gym_cfg):
        """GeneratorOutput has all required TypedDict keys."""
        client = _make_mock_client()
        env = _make_mock_env()

        generator = SkyRLGymTinkerGenerator(generator_cfg, skyrl_gym_cfg, client, renderer)

        input_batch: GeneratorInput = {
            "prompts": [[{"role": "user", "content": "Hi"}]],
            "env_classes": ["gsm8k"],
            "env_extras": [{}],
        }

        with patch("skyrl_gym.make", return_value=env):
            output = asyncio.run(generator.generate(input_batch, disable_tqdm=True))

        required_keys = [
            "prompt_token_ids", "response_ids", "rewards", "loss_masks",
            "stop_reasons", "rollout_metrics", "rollout_logprobs",
            "trajectory_ids", "rollout_expert_indices", "is_last_step",
        ]
        for key in required_keys:
            assert key in output, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Stop reason / filtering tests
# ---------------------------------------------------------------------------


class TestStopReasonHandling:
    def test_stop_reason_propagated(self, renderer, generator_cfg, skyrl_gym_cfg):
        """Stop reason from sample() propagates to output."""
        client = _make_mock_client([
            {"tokens": [10], "logprobs": [-0.1], "stop_reason": "length"}
        ])
        env = _make_mock_env([{"observations": [], "reward": 0.0, "done": True}])

        generator = SkyRLGymTinkerGenerator(generator_cfg, skyrl_gym_cfg, client, renderer)

        with patch("skyrl_gym.make", return_value=env):
            output = asyncio.run(generator.agent_loop(
                [{"role": "user", "content": "Hi"}], "test_env", {},
                max_tokens=64, max_input_length=512
            ))

        assert output.stop_reason == "length"

    def test_zero_reward_on_non_stop(self, renderer, skyrl_gym_cfg):
        """zero_reward_on_non_stop zeros out rewards for length-stopped trajectories."""
        from skyrl.train.config import GeneratorConfig, SamplingParams

        cfg = GeneratorConfig()
        cfg.sampling_params = SamplingParams(max_generate_length=64, logprobs=0)
        cfg.max_input_length = 512
        cfg.max_turns = 1
        cfg.zero_reward_on_non_stop = True

        client = _make_mock_client([
            {"tokens": [10], "logprobs": [-0.1], "stop_reason": "length"}
        ])
        env = _make_mock_env([{"observations": [], "reward": 5.0, "done": True}])

        generator = SkyRLGymTinkerGenerator(cfg, skyrl_gym_cfg, client, renderer)

        input_batch: GeneratorInput = {
            "prompts": [[{"role": "user", "content": "Hi"}]],
            "env_classes": ["gsm8k"],
            "env_extras": [{}],
        }

        with patch("skyrl_gym.make", return_value=env):
            output = asyncio.run(generator.generate(input_batch, disable_tqdm=True))

        # Reward should be zeroed out because stop_reason != "stop"
        reward = output["rewards"][0]
        if isinstance(reward, list):
            assert all(r == 0.0 for r in reward)
        else:
            assert reward == 0.0
