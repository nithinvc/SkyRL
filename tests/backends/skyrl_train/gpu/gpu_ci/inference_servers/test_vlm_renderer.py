"""
GPU integration tests for VLLMRenderer with a real vLLM server.

Tests that VLLMRenderer correctly renders multi-modal ModelInputs
using a real Qwen3-VL model and the /v1/chat/completions/render endpoint.

# Run with:
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_vlm_renderer.py -m vllm -v
"""

from __future__ import annotations

import asyncio
import base64
import io

import pytest
from PIL import Image

from skyrl.backends.renderer import VLLMRenderer
from skyrl.tinker.types import (
    EncodedTextChunk,
    ImageChunk,
    ModelInput,
)
from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

MODEL_QWEN3_VL = "Qwen/Qwen3-VL-2B-Instruct"
TP_SIZE = 1


def _get_test_config(num_inference_engines: int) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL_QWEN3_VL
    cfg.trainer.critic.model.path = ""
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.placement.policy_num_gpus_per_node = TP_SIZE * num_inference_engines
    cfg.generator.async_engine = True
    cfg.generator.inference_engine.num_engines = num_inference_engines
    cfg.generator.inference_engine.tensor_parallel_size = TP_SIZE
    cfg.generator.run_engines_locally = True
    cfg.generator.inference_engine.served_model_name = MODEL_QWEN3_VL
    cfg.generator.sampling_params.max_generate_length = 256
    return cfg


def _make_tiny_jpeg_bytes() -> bytes:
    """Create a minimal 8x8 red JPEG image and return raw bytes."""
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _tokenize_text(client, text: str) -> list[int]:
    """Tokenize text using the client's tokenizer."""
    return client.tokenizer.encode(text, add_special_tokens=False)


@pytest.mark.vllm
def test_vlm_renderer_single_image(module_scoped_ray_init_fixture):
    """VLLMRenderer renders a single image ModelInput with correct tokens and placeholders."""
    engines = None
    try:
        cfg = _get_test_config(num_inference_engines=1)
        engines = InferenceEngineState.create(
            cfg=cfg,
            use_local=True,
            backend="vllm",
            model=MODEL_QWEN3_VL,
            sleep_level=1,
            engine_init_kwargs={
                "max_model_len": 4096,
                "limit_mm_per_prompt": {"image": 1, "video": 0},
            },
            use_new_inference_servers=True,
        )
        client = engines.client

        # Build ModelInput with text + image + text
        img_bytes = _make_tiny_jpeg_bytes()
        b64_img = base64.b64encode(img_bytes)  # pydantic Base64Bytes input

        text_before_tokens = _tokenize_text(client, "Describe this image:")
        text_after_tokens = _tokenize_text(client, " Be concise.")

        mi = ModelInput(
            chunks=[
                EncodedTextChunk(tokens=text_before_tokens),
                ImageChunk(data=b64_img, format="jpeg"),
                EncodedTextChunk(tokens=text_after_tokens),
            ]
        )

        renderer = VLLMRenderer(client=client, model_name=MODEL_QWEN3_VL)
        results = renderer([mi])

        assert len(results) == 1
        result = results[0]

        # prompt_ids should contain text tokens + image placeholder tokens + text tokens
        assert len(result.prompt_ids) > len(text_before_tokens) + len(text_after_tokens)

        # Should have exactly one image placeholder
        assert result.multi_modal_placeholders is not None
        assert len(result.multi_modal_placeholders) == 1

        ph = result.multi_modal_placeholders[0]
        assert ph.offset == len(text_before_tokens)
        assert ph.length > 0

        # Total tokens = text_before + placeholder + text_after
        assert len(result.prompt_ids) == len(text_before_tokens) + ph.length + len(text_after_tokens)

        # Text tokens should be preserved at correct positions
        assert result.prompt_ids[: len(text_before_tokens)] == text_before_tokens
        assert result.prompt_ids[ph.offset + ph.length :] == text_after_tokens

        # kwargs_data should be captured (Qwen3-VL render returns it)
        if result.multi_modal_kwargs is not None:
            assert "image" in result.multi_modal_kwargs
            assert len(result.multi_modal_kwargs["image"]) == 1

    finally:
        if engines is not None:
            engines.close()


@pytest.mark.vllm
def test_vlm_renderer_text_only_no_render_call(module_scoped_ray_init_fixture):
    """VLLMRenderer with text-only input should not make any render calls."""
    engines = None
    try:
        cfg = _get_test_config(num_inference_engines=1)
        engines = InferenceEngineState.create(
            cfg=cfg,
            use_local=True,
            backend="vllm",
            model=MODEL_QWEN3_VL,
            sleep_level=1,
            engine_init_kwargs={
                "max_model_len": 4096,
                "limit_mm_per_prompt": {"image": 1, "video": 0},
            },
            use_new_inference_servers=True,
        )
        client = engines.client

        tokens = _tokenize_text(client, "Hello, world!")
        mi = ModelInput(chunks=[EncodedTextChunk(tokens=tokens)])

        renderer = VLLMRenderer(client=client, model_name=MODEL_QWEN3_VL)
        results = renderer([mi])

        assert len(results) == 1
        assert results[0].prompt_ids == tokens
        assert results[0].multi_modal_placeholders is None
        assert results[0].multi_modal_kwargs is None

    finally:
        if engines is not None:
            engines.close()


@pytest.mark.vllm
def test_vlm_renderer_kwargs_data_decodes_to_tensors(module_scoped_ray_init_fixture):
    """kwargs_data from render response decodes to pixel_values and image_grid_thw tensors."""
    engines = None
    try:
        cfg = _get_test_config(num_inference_engines=1)
        engines = InferenceEngineState.create(
            cfg=cfg,
            use_local=True,
            backend="vllm",
            model=MODEL_QWEN3_VL,
            sleep_level=1,
            engine_init_kwargs={
                "max_model_len": 4096,
                "limit_mm_per_prompt": {"image": 1, "video": 0},
            },
            use_new_inference_servers=True,
        )
        client = engines.client

        img_bytes = _make_tiny_jpeg_bytes()
        b64_img = base64.b64encode(img_bytes)

        mi = ModelInput(
            chunks=[
                ImageChunk(data=b64_img, format="jpeg"),
            ]
        )

        renderer = VLLMRenderer(client=client, model_name=MODEL_QWEN3_VL)
        results = renderer([mi])
        result = results[0]

        # Skip this test if kwargs_data is not returned (depends on vLLM version)
        if result.multi_modal_kwargs is None:
            pytest.skip("vLLM server did not return kwargs_data")

        # Decode kwargs_data and verify it contains tensors
        from vllm.entrypoints.serve.disagg.mm_serde import decode_mm_kwargs_item

        for b64_str in result.multi_modal_kwargs["image"]:
            item = decode_mm_kwargs_item(b64_str)
            data = item.get_data()

            assert "pixel_values" in data, f"Expected pixel_values in kwargs_data, got keys: {list(data.keys())}"
            assert "image_grid_thw" in data, f"Expected image_grid_thw in kwargs_data, got keys: {list(data.keys())}"

            import torch

            pv = data["pixel_values"]
            thw = data["image_grid_thw"]
            assert isinstance(pv, torch.Tensor), f"pixel_values should be a tensor, got {type(pv)}"
            assert isinstance(thw, torch.Tensor), f"image_grid_thw should be a tensor, got {type(thw)}"
            assert pv.ndim >= 1 and pv.shape[0] > 0, f"pixel_values should be non-empty, got shape {pv.shape}"
            assert thw.ndim == 1 and thw.shape[0] == 3, f"per-item image_grid_thw should be [3], got shape {thw.shape}"

    finally:
        if engines is not None:
            engines.close()
