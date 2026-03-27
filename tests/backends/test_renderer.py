"""CPU unit tests for VLLMRenderer with mocked RemoteInferenceClient."""

from __future__ import annotations

import base64
import logging
from unittest.mock import AsyncMock

import pytest

from skyrl.backends.renderer import VLLMRenderer, render_model_input
from skyrl.tinker.types import (
    EncodedTextChunk,
    ImageAssetPointerChunk,
    ImageChunk,
    ModelInput,
)


def _make_client(render_response=None):
    """Create a mock RemoteInferenceClient with a configurable render_chat_completion."""
    client = AsyncMock()
    if render_response is not None:
        client.render_chat_completion.return_value = render_response
    return client


def _render_response(token_ids, image_placeholders, kwargs_data=None):
    """Build a mock /v1/chat/completions/render response."""
    features = {
        "mm_placeholders": {
            "image": [{"offset": ph[0], "length": ph[1]} for ph in image_placeholders],
        },
    }
    if kwargs_data is not None:
        features["kwargs_data"] = {"image": kwargs_data}
    return {
        "token_ids": token_ids,
        "features": features,
    }


# ---------------------------------------------------------------------------
# Text-only: no HTTP calls, tokens concatenated
# ---------------------------------------------------------------------------


class TestTextOnly:
    def test_text_only_no_render_call(self):
        client = _make_client()
        renderer = VLLMRenderer(client=client, model_name="test-model")

        mi = ModelInput(
            chunks=[
                EncodedTextChunk(tokens=[1, 2, 3]),
                EncodedTextChunk(tokens=[4, 5]),
            ]
        )
        results = renderer([mi])

        assert len(results) == 1
        assert results[0].prompt_ids == [1, 2, 3, 4, 5]
        assert results[0].multi_modal_placeholders is None
        assert results[0].multi_modal_kwargs is None
        client.render_chat_completion.assert_not_called()

    def test_text_only_matches_free_function(self):
        """VLLMRenderer text-only path should match the existing render_model_input."""
        client = _make_client()
        renderer = VLLMRenderer(client=client, model_name="test-model")

        mi = ModelInput(
            chunks=[
                EncodedTextChunk(tokens=[10, 20]),
                EncodedTextChunk(tokens=[30]),
            ]
        )
        vllm_result = renderer([mi])
        free_result = render_model_input([mi])

        assert vllm_result[0].prompt_ids == free_result[0].prompt_ids


# ---------------------------------------------------------------------------
# Single image
# ---------------------------------------------------------------------------


class TestSingleImage:
    def test_single_image_render_payload_and_extraction(self):
        # Base64Bytes: pydantic accepts base64-encoded input, stores as raw bytes
        raw_bytes = b"\x89PNG_fake"
        b64_input = base64.b64encode(raw_bytes)  # bytes that pydantic will decode
        # After pydantic decodes, chunk.data == raw_bytes; renderer re-encodes for the URL
        re_encoded_b64 = base64.b64encode(raw_bytes).decode("ascii")

        # Render response: chat template tokens [100, 101] + image placeholders [200, 201, 202] + tail [102]
        response = _render_response(
            token_ids=[100, 101, 200, 201, 202, 102],
            image_placeholders=[(2, 3)],  # offset=2, length=3
        )
        client = _make_client(response)
        renderer = VLLMRenderer(client=client, model_name="my-vlm")

        mi = ModelInput(
            chunks=[
                ImageChunk(data=b64_input, format="png"),
            ]
        )
        results = renderer([mi])

        # Check render was called with correct payload
        call_args = client.render_chat_completion.call_args
        payload = call_args[0][0]  # first positional arg
        body = payload["json"]
        assert body["model"] == "my-vlm"
        assert len(body["messages"]) == 1
        content = body["messages"][0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == f"data:image/png;base64,{re_encoded_b64}"

        # Check output
        result = results[0]
        assert result.prompt_ids == [200, 201, 202]
        assert len(result.multi_modal_placeholders) == 1
        assert result.multi_modal_placeholders[0].offset == 0
        assert result.multi_modal_placeholders[0].length == 3


# ---------------------------------------------------------------------------
# Mixed text + image: correct assembly order and placeholder offsets
# ---------------------------------------------------------------------------


class TestMixedTextImage:
    def test_text_image_text_assembly(self):
        response = _render_response(
            token_ids=[100, 101, 500, 501, 502, 503, 102],
            image_placeholders=[(2, 4)],  # offset=2, length=4
        )
        client = _make_client(response)
        renderer = VLLMRenderer(client=client, model_name="test-model")

        mi = ModelInput(
            chunks=[
                EncodedTextChunk(tokens=[1, 2, 3]),
                ImageChunk(data=b"fake", format="jpeg"),
                EncodedTextChunk(tokens=[7, 8]),
            ]
        )
        results = renderer([mi])
        result = results[0]

        # prompt_ids = text[1,2,3] + placeholder[500,501,502,503] + text[7,8]
        assert result.prompt_ids == [1, 2, 3, 500, 501, 502, 503, 7, 8]

        # Placeholder offset should be 3 (after the first text chunk)
        assert len(result.multi_modal_placeholders) == 1
        assert result.multi_modal_placeholders[0].offset == 3
        assert result.multi_modal_placeholders[0].length == 4


# ---------------------------------------------------------------------------
# Multiple images: all placeholders tracked, offsets adjusted
# ---------------------------------------------------------------------------


class TestMultipleImages:
    def test_two_images_with_text_between(self):
        # Render response: template [100] + img1 placeholders [300,301] + img2 placeholders [400,401,402] + tail [101]
        response = _render_response(
            token_ids=[100, 300, 301, 400, 401, 402, 101],
            image_placeholders=[(1, 2), (3, 3)],  # img1: offset=1 len=2, img2: offset=3 len=3
        )
        client = _make_client(response)
        renderer = VLLMRenderer(client=client, model_name="test-model")

        mi = ModelInput(
            chunks=[
                EncodedTextChunk(tokens=[10, 11]),
                ImageChunk(data=b"img1", format="png"),
                EncodedTextChunk(tokens=[20]),
                ImageChunk(data=b"img2", format="jpeg"),
                EncodedTextChunk(tokens=[30, 31]),
            ]
        )
        results = renderer([mi])
        result = results[0]

        # Assembly: [10,11] + [300,301] + [20] + [400,401,402] + [30,31]
        assert result.prompt_ids == [10, 11, 300, 301, 20, 400, 401, 402, 30, 31]

        assert len(result.multi_modal_placeholders) == 2
        # First image placeholder at offset 2 (after [10,11])
        assert result.multi_modal_placeholders[0].offset == 2
        assert result.multi_modal_placeholders[0].length == 2
        # Second image placeholder at offset 5 (after [10,11,300,301,20])
        assert result.multi_modal_placeholders[1].offset == 5
        assert result.multi_modal_placeholders[1].length == 3


# ---------------------------------------------------------------------------
# ImageAssetPointerChunk: URL passed through
# ---------------------------------------------------------------------------


class TestImageAssetPointer:
    def test_asset_pointer_url_passed_through(self):
        response = _render_response(
            token_ids=[100, 600, 601, 101],
            image_placeholders=[(1, 2)],
        )
        client = _make_client(response)
        renderer = VLLMRenderer(client=client, model_name="test-model")

        mi = ModelInput(
            chunks=[
                ImageAssetPointerChunk(
                    format="png",
                    location="https://storage.example.com/images/cat.png",
                ),
            ]
        )
        results = renderer([mi])

        # Verify the URL was passed through as-is
        call_args = client.render_chat_completion.call_args
        content = call_args[0][0]["json"]["messages"][0]["content"]
        assert content[0]["image_url"]["url"] == "https://storage.example.com/images/cat.png"

        assert results[0].prompt_ids == [600, 601]


# ---------------------------------------------------------------------------
# expected_tokens mismatch: warning logged, no error
# ---------------------------------------------------------------------------


class TestExpectedTokensWarning:
    def test_mismatch_logs_warning(self, caplog):
        response = _render_response(
            token_ids=[100, 700, 701, 702, 101],
            image_placeholders=[(1, 3)],  # actual length=3
        )
        client = _make_client(response)
        renderer = VLLMRenderer(client=client, model_name="test-model")

        mi = ModelInput(
            chunks=[
                ImageChunk(data=b"fake", format="png", expected_tokens=5),  # expected 5, got 3
            ]
        )

        with caplog.at_level(logging.WARNING):
            results = renderer([mi])

        # Should still succeed
        assert results[0].prompt_ids == [700, 701, 702]

        # Warning should be logged
        assert any("expected_tokens=5" in record.message and "3" in record.message for record in caplog.records)

    def test_matching_expected_tokens_no_warning(self, caplog):
        response = _render_response(
            token_ids=[100, 700, 701, 702, 101],
            image_placeholders=[(1, 3)],
        )
        client = _make_client(response)
        renderer = VLLMRenderer(client=client, model_name="test-model")

        mi = ModelInput(
            chunks=[
                ImageChunk(data=b"fake", format="png", expected_tokens=3),  # matches
            ]
        )

        with caplog.at_level(logging.WARNING):
            results = renderer([mi])

        assert results[0].prompt_ids == [700, 701, 702]
        assert not any("expected_tokens" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Render error: RuntimeError propagates
# ---------------------------------------------------------------------------


class TestRenderError:
    def test_render_api_error_propagates(self):
        client = _make_client()
        client.render_chat_completion.side_effect = RuntimeError("render endpoint failed")
        renderer = VLLMRenderer(client=client, model_name="test-model")

        mi = ModelInput(
            chunks=[
                ImageChunk(data=b"fake", format="png"),
            ]
        )

        with pytest.raises(RuntimeError, match="render endpoint failed"):
            renderer([mi])

    def test_placeholder_count_mismatch_raises(self):
        # Response has 0 placeholders but we sent 1 image
        response = _render_response(
            token_ids=[100, 101],
            image_placeholders=[],
        )
        client = _make_client(response)
        renderer = VLLMRenderer(client=client, model_name="test-model")

        mi = ModelInput(
            chunks=[
                ImageChunk(data=b"fake", format="png"),
            ]
        )

        with pytest.raises(RuntimeError, match="Expected 1 image placeholders, got 0"):
            renderer([mi])


# ---------------------------------------------------------------------------
# kwargs_data capture
# ---------------------------------------------------------------------------


class TestKwargsDataCapture:
    def test_kwargs_data_captured_in_multi_modal_kwargs(self):
        response = _render_response(
            token_ids=[100, 200, 201, 101],
            image_placeholders=[(1, 2)],
            kwargs_data=["base64_encoded_pixel_values"],
        )
        client = _make_client(response)
        renderer = VLLMRenderer(client=client, model_name="test-model")

        mi = ModelInput(
            chunks=[
                ImageChunk(data=b"fake", format="png"),
            ]
        )
        results = renderer([mi])

        assert results[0].multi_modal_kwargs is not None
        assert "image" in results[0].multi_modal_kwargs
        assert results[0].multi_modal_kwargs["image"] == ["base64_encoded_pixel_values"]

    def test_kwargs_data_absent_means_none(self):
        response = _render_response(
            token_ids=[100, 200, 201, 101],
            image_placeholders=[(1, 2)],
            kwargs_data=None,  # No kwargs_data in response
        )
        client = _make_client(response)
        renderer = VLLMRenderer(client=client, model_name="test-model")

        mi = ModelInput(
            chunks=[
                ImageChunk(data=b"fake", format="png"),
            ]
        )
        results = renderer([mi])

        assert results[0].multi_modal_kwargs is None

    def test_multiple_images_kwargs_data(self):
        response = _render_response(
            token_ids=[100, 200, 201, 300, 301, 302, 101],
            image_placeholders=[(1, 2), (3, 3)],
            kwargs_data=["kd_image_0", "kd_image_1"],
        )
        client = _make_client(response)
        renderer = VLLMRenderer(client=client, model_name="test-model")

        mi = ModelInput(
            chunks=[
                ImageChunk(data=b"img1", format="png"),
                ImageChunk(data=b"img2", format="jpeg"),
            ]
        )
        results = renderer([mi])

        assert results[0].multi_modal_kwargs == {"image": ["kd_image_0", "kd_image_1"]}
