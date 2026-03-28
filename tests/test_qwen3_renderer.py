"""
Layer 1: Renderer unit tests for Qwen3Renderer and Qwen3VLRenderer.

Uses a real Qwen3 tokenizer (CPU-only, no GPU needed).
Tests text rendering, multi-turn extension, image handling, tool call parsing, and roundtrip.

Run with:
    uv run --extra dev --extra fsdp pytest tests/test_qwen3_renderer.py -v
"""

from __future__ import annotations

import io

import pytest
from PIL import Image
from transformers import AutoTokenizer

from skyrl.tinker.types import EncodedTextChunk, ImageChunk
from skyrl.train.renderers.base import (
    Message,
    TextPart,
    ThinkingPart,
    ImagePart,
    ToolCall,
    get_text_content,
    image_to_chunk,
)
from skyrl.train.renderers.qwen3 import Qwen3Renderer, Qwen3VLRenderer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-8B"
VL_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


@pytest.fixture(scope="module")
def vl_tokenizer():
    return AutoTokenizer.from_pretrained(VL_MODEL_NAME, use_fast=True)


@pytest.fixture(scope="module")
def vl_image_processor():
    from skyrl.train.renderers.image_utils import get_image_processor

    return get_image_processor(VL_MODEL_NAME)


def _make_tiny_image(width=8, height=8, color=(255, 0, 0)) -> Image.Image:
    """Create a minimal PIL image for testing."""
    return Image.new("RGB", (width, height), color=color)


def _count_tokens(model_input) -> int:
    """Count total tokens in a ModelInput."""
    total = 0
    for chunk in model_input.chunks:
        if isinstance(chunk, EncodedTextChunk):
            total += len(chunk.tokens)
        elif isinstance(chunk, ImageChunk) and chunk.expected_tokens is not None:
            total += chunk.expected_tokens
    return total


def _flatten_text_tokens(model_input) -> list[int]:
    """Extract all text tokens from a ModelInput (skip image chunks)."""
    tokens = []
    for chunk in model_input.chunks:
        if isinstance(chunk, EncodedTextChunk):
            tokens.extend(chunk.tokens)
    return tokens


# ---------------------------------------------------------------------------
# Qwen3Renderer: Text rendering
# ---------------------------------------------------------------------------


class TestQwen3TextRendering:
    def test_single_user_message(self, tokenizer):
        renderer = Qwen3Renderer(tokenizer)
        messages = [Message(role="user", content="Hello!")]
        model_input = renderer.build_generation_prompt(messages)

        # Should produce: <|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n
        tokens = _flatten_text_tokens(model_input)
        decoded = tokenizer.decode(tokens)
        assert "<|im_start|>user" in decoded
        assert "Hello!" in decoded
        assert "<|im_end|>" in decoded
        assert "<|im_start|>assistant" in decoded

    def test_system_and_user(self, tokenizer):
        renderer = Qwen3Renderer(tokenizer)
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi"),
        ]
        model_input = renderer.build_generation_prompt(messages)
        tokens = _flatten_text_tokens(model_input)
        decoded = tokenizer.decode(tokens)
        assert "You are helpful." in decoded
        assert "Hi" in decoded

    def test_multi_turn_conversation(self, tokenizer):
        renderer = Qwen3Renderer(tokenizer)
        messages = [
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="4"),
            Message(role="user", content="And 3+3?"),
        ]
        model_input = renderer.build_generation_prompt(messages)
        tokens = _flatten_text_tokens(model_input)
        decoded = tokenizer.decode(tokens)
        assert "What is 2+2?" in decoded
        assert "4" in decoded
        assert "And 3+3?" in decoded

    def test_stop_sequences(self, tokenizer):
        renderer = Qwen3Renderer(tokenizer)
        stops = renderer.get_stop_sequences()
        assert len(stops) == 1
        assert isinstance(stops[0], int)
        # Should be the <|im_end|> token
        decoded = tokenizer.decode([stops[0]])
        assert "<|im_end|>" in decoded


# ---------------------------------------------------------------------------
# Extension property
# ---------------------------------------------------------------------------


class TestExtensionProperty:
    """Test the sequence extension property.

    The extension property means: the rendered conversation (without generation
    suffix) at turn N is a prefix of the rendered conversation at turn N+1.
    The generation suffix ("<|im_start|>assistant\\n") moves position each turn,
    so we strip it before comparing.
    """

    @staticmethod
    def _render_without_suffix(renderer, messages) -> list[int]:
        """Render messages into tokens, stripping the trailing generation suffix."""
        # Build full prompt (includes gen suffix at end)
        model_input = renderer.build_generation_prompt(messages)
        all_tokens = _flatten_text_tokens(model_input)

        # The gen suffix is the assistant role header appended at the end.
        # Compute it so we can strip it.
        from skyrl.train.renderers.base import RenderContext

        suffix_ctx = RenderContext(idx=len(messages), is_last=True, prev_message=messages[-1] if messages else None)
        suffix_tokens = renderer._get_generation_suffix("assistant", suffix_ctx)
        if suffix_tokens and all_tokens[-len(suffix_tokens) :] == suffix_tokens:
            return all_tokens[: -len(suffix_tokens)]
        return all_tokens

    def test_extension_with_strip_thinking_false(self, tokenizer):
        """With strip_thinking=False, rendered conversation grows as a prefix."""
        renderer = Qwen3Renderer(tokenizer, strip_thinking_from_history=False)
        assert renderer.has_extension_property

        messages_2 = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        messages_3 = messages_2 + [
            Message(role="user", content="How are you?"),
        ]

        conv_2 = self._render_without_suffix(renderer, messages_2)
        conv_3 = self._render_without_suffix(renderer, messages_3)

        assert len(conv_3) > len(conv_2)
        assert conv_3[: len(conv_2)] == conv_2

    def test_extension_with_thinking_blocks(self, tokenizer):
        """Extension holds even when assistant messages have thinking blocks (strip=False)."""
        renderer = Qwen3Renderer(tokenizer, strip_thinking_from_history=False)

        messages_2 = [
            Message(role="user", content="Solve 2+2"),
            Message(
                role="assistant",
                content=[
                    ThinkingPart(type="thinking", thinking="Let me think..."),
                    TextPart(type="text", text="4"),
                ],
            ),
        ]
        messages_3 = messages_2 + [
            Message(role="user", content="And 3+3?"),
        ]

        conv_2 = self._render_without_suffix(renderer, messages_2)
        conv_3 = self._render_without_suffix(renderer, messages_3)

        assert len(conv_3) > len(conv_2)
        assert conv_3[: len(conv_2)] == conv_2

    def test_no_extension_with_strip_thinking_true(self, tokenizer):
        renderer = Qwen3Renderer(tokenizer, strip_thinking_from_history=True)
        assert not renderer.has_extension_property

    def test_extension_breaks_when_thinking_stripped(self, tokenizer):
        """With strip_thinking=True, the extension property breaks for thinking content."""
        renderer = Qwen3Renderer(tokenizer, strip_thinking_from_history=True)

        messages_2 = [
            Message(role="user", content="Solve 2+2"),
            Message(
                role="assistant",
                content=[
                    ThinkingPart(type="thinking", thinking="reasoning"),
                    TextPart(type="text", text="4"),
                ],
            ),
        ]
        messages_3 = messages_2 + [
            Message(role="user", content="And 3+3?"),
        ]

        conv_2 = self._render_without_suffix(renderer, messages_2)
        conv_3 = self._render_without_suffix(renderer, messages_3)

        # Extension should break: conv_3 strips thinking from the historical assistant msg
        assert conv_3[: len(conv_2)] != conv_2


# ---------------------------------------------------------------------------
# Thinking blocks
# ---------------------------------------------------------------------------


class TestThinkingBlocks:
    def test_thinking_stripped_from_history(self, tokenizer):
        """With strip_thinking=True, historical assistant thinking blocks are removed."""
        renderer = Qwen3Renderer(tokenizer, strip_thinking_from_history=True)

        messages = [
            Message(role="user", content="Solve 2+2"),
            Message(
                role="assistant",
                content=[
                    ThinkingPart(type="thinking", thinking="reasoning here"),
                    TextPart(type="text", text="4"),
                ],
            ),
            Message(role="user", content="And 3+3?"),
        ]
        tokens = _flatten_text_tokens(renderer.build_generation_prompt(messages))
        decoded = tokenizer.decode(tokens)
        assert "reasoning here" not in decoded
        assert "4" in decoded

    def test_thinking_preserved_in_last_message(self, tokenizer):
        """Even with strip_thinking=True, the last assistant message keeps thinking."""
        renderer = Qwen3Renderer(tokenizer, strip_thinking_from_history=True)

        messages = [
            Message(role="user", content="Solve 2+2"),
            Message(
                role="assistant",
                content=[
                    ThinkingPart(type="thinking", thinking="reasoning here"),
                    TextPart(type="text", text="4"),
                ],
            ),
        ]
        tokens = _flatten_text_tokens(renderer.build_generation_prompt(messages))
        decoded = tokenizer.decode(tokens)
        assert "reasoning here" in decoded


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------


class TestToolCallParsing:
    def test_parse_tool_call_response(self, tokenizer):
        renderer = Qwen3Renderer(tokenizer)
        stop_token = renderer.get_stop_sequences()[0]

        tool_call_text = '<tool_call>\n{"name": "calc_score", "arguments": {"answer": "42"}}\n</tool_call>'
        response_tokens = tokenizer.encode(tool_call_text + "<|im_end|>", add_special_tokens=False)

        message, success = renderer.parse_response(response_tokens)
        assert success
        assert "tool_calls" in message
        assert len(message["tool_calls"]) == 1
        tc = message["tool_calls"][0]
        assert tc.function.name == "calc_score"
        assert '"answer"' in tc.function.arguments

    def test_parse_thinking_then_tool_call(self, tokenizer):
        renderer = Qwen3Renderer(tokenizer)

        text = '<think>Let me solve this</think>\n<tool_call>\n{"name": "search", "arguments": {"q": "test"}}\n</tool_call>'
        response_tokens = tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)

        message, success = renderer.parse_response(response_tokens)
        assert success
        assert isinstance(message["content"], list)
        types = [p["type"] for p in message["content"]]
        assert "thinking" in types
        assert "tool_call" in types

    def test_parse_plain_text_response(self, tokenizer):
        renderer = Qwen3Renderer(tokenizer)

        text = "The answer is 42."
        response_tokens = tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)

        message, success = renderer.parse_response(response_tokens)
        assert success
        assert isinstance(message["content"], str)
        assert message["content"] == text

    def test_parse_missing_stop_token(self, tokenizer):
        renderer = Qwen3Renderer(tokenizer)

        text = "Incomplete response"
        response_tokens = tokenizer.encode(text, add_special_tokens=False)

        message, success = renderer.parse_response(response_tokens)
        assert not success
        assert "Incomplete response" in get_text_content(message)


# ---------------------------------------------------------------------------
# Roundtrip: render → sample → parse
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_render_parse_roundtrip(self, tokenizer):
        """The text from parse_response should match what was rendered."""
        renderer = Qwen3Renderer(tokenizer)

        messages = [
            Message(role="user", content="What is 2+2?"),
        ]
        model_input = renderer.build_generation_prompt(messages)

        # Simulate model generating "4<|im_end|>"
        response_text = "4"
        stop_token = renderer.get_stop_sequences()[0]
        response_tokens = tokenizer.encode(response_text, add_special_tokens=False) + [stop_token]

        message, success = renderer.parse_response(response_tokens)
        assert success
        assert get_text_content(message) == response_text


# ---------------------------------------------------------------------------
# Qwen3VLRenderer: Image handling
# ---------------------------------------------------------------------------


class TestQwen3VLImageHandling:
    def test_image_chunk_creation(self, vl_image_processor):
        """image_to_chunk creates an ImageChunk with expected_tokens."""
        img = _make_tiny_image()
        chunk = image_to_chunk(img, vl_image_processor)
        assert isinstance(chunk, ImageChunk)
        assert chunk.format == "jpeg"
        assert chunk.expected_tokens is not None
        assert chunk.expected_tokens > 0

    def test_vl_render_with_image(self, vl_tokenizer, vl_image_processor):
        """VL renderer produces ImageChunk for image content."""
        renderer = Qwen3VLRenderer(vl_tokenizer, vl_image_processor)

        img = _make_tiny_image()
        messages = [
            Message(
                role="user",
                content=[
                    ImagePart(type="image", image=img),
                    TextPart(type="text", text="What is this?"),
                ],
            ),
        ]
        model_input = renderer.build_generation_prompt(messages)

        # Should contain at least one ImageChunk
        image_chunks = [c for c in model_input.chunks if isinstance(c, ImageChunk)]
        assert len(image_chunks) == 1
        assert image_chunks[0].expected_tokens is not None

        # Text chunks should contain vision tokens
        text_tokens = _flatten_text_tokens(model_input)
        decoded = vl_tokenizer.decode(text_tokens)
        assert "<|vision_start|>" in decoded
        assert "<|vision_end|>" in decoded
        assert "What is this?" in decoded

    def test_vl_text_only_no_image_chunks(self, vl_tokenizer, vl_image_processor):
        """VL renderer with text-only content produces no ImageChunks."""
        renderer = Qwen3VLRenderer(vl_tokenizer, vl_image_processor)

        messages = [Message(role="user", content="Hello")]
        model_input = renderer.build_generation_prompt(messages)

        image_chunks = [c for c in model_input.chunks if isinstance(c, ImageChunk)]
        assert len(image_chunks) == 0

    def test_vl_multi_turn_with_image_extension(self, vl_tokenizer, vl_image_processor):
        """Extension property holds for VL renderer with images (text tokens only)."""
        renderer = Qwen3VLRenderer(vl_tokenizer, vl_image_processor, strip_thinking_from_history=False)
        assert renderer.has_extension_property

        img = _make_tiny_image()
        messages_2 = [
            Message(
                role="user",
                content=[
                    ImagePart(type="image", image=img),
                    TextPart(type="text", text="Describe this."),
                ],
            ),
            Message(role="assistant", content="A red square."),
        ]
        messages_3 = messages_2 + [
            Message(role="user", content="What color?"),
        ]

        # Strip generation suffix before comparing (same approach as text tests)
        conv_2 = TestExtensionProperty._render_without_suffix(renderer, messages_2)
        conv_3 = TestExtensionProperty._render_without_suffix(renderer, messages_3)
        assert len(conv_3) > len(conv_2)
        assert conv_3[: len(conv_2)] == conv_2

    def test_vl_multiple_images(self, vl_tokenizer, vl_image_processor):
        """VL renderer handles multiple images in a single message."""
        renderer = Qwen3VLRenderer(vl_tokenizer, vl_image_processor)

        img1 = _make_tiny_image(color=(255, 0, 0))
        img2 = _make_tiny_image(color=(0, 255, 0))

        messages = [
            Message(
                role="user",
                content=[
                    ImagePart(type="image", image=img1),
                    ImagePart(type="image", image=img2),
                    TextPart(type="text", text="Compare these."),
                ],
            ),
        ]
        model_input = renderer.build_generation_prompt(messages)

        image_chunks = [c for c in model_input.chunks if isinstance(c, ImageChunk)]
        assert len(image_chunks) == 2

    def test_vl_parse_response_same_as_text(self, vl_tokenizer, vl_image_processor):
        """VL renderer's parse_response works the same as text renderer."""
        renderer = Qwen3VLRenderer(vl_tokenizer, vl_image_processor)
        stop_token = renderer.get_stop_sequences()[0]

        text = "This is a red square."
        response_tokens = vl_tokenizer.encode(text, add_special_tokens=False) + [stop_token]

        message, success = renderer.parse_response(response_tokens)
        assert success
        assert get_text_content(message) == text


# ---------------------------------------------------------------------------
# image_to_chunk edge cases
# ---------------------------------------------------------------------------


class TestImageToChunk:
    def test_rgba_image_converted(self, vl_image_processor):
        """RGBA images are converted to RGB before encoding."""
        img = Image.new("RGBA", (8, 8), color=(255, 0, 0, 128))
        chunk = image_to_chunk(img, vl_image_processor)
        assert chunk.format == "jpeg"

    def test_palette_image_converted(self, vl_image_processor):
        """Palette (P mode) images are converted to RGB."""
        img = Image.new("P", (8, 8))
        chunk = image_to_chunk(img, vl_image_processor)
        assert chunk.format == "jpeg"
