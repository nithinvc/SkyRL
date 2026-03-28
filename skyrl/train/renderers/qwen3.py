"""
Qwen3 family renderers — text and vision-language models.

Includes:
- Qwen3Renderer: Base Qwen3 with thinking enabled
- Qwen3VLRenderer: Vision-language Qwen3 with thinking

Copied from tinker-cookbook/tinker_cookbook/renderers/qwen3.py (trimmed).
"""

import json
from typing import cast

from skyrl.tinker.types import EncodedTextChunk, ModelInputChunk
from skyrl.train.renderers.base import (
    ImagePart,
    ImageProcessorProtocol,
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    TextPart,
    ToolCall,
    UnparsedToolCall,
    _tool_call_payload,
    image_to_chunk,
    parse_content_blocks,
    parse_response_for_stop_token,
    remove_thinking,
)
from skyrl.train.renderers.image_utils import ImageProcessor


def _merge_consecutive_text_parts(
    chunks: list[ImagePart | TextPart],
) -> list[ImagePart | TextPart]:
    """Merge consecutive TextParts into single parts.

    This ensures text is tokenized as a single string, matching HuggingFace's
    apply_chat_template behavior which tokenizes the full rendered string at once.
    """
    if not chunks:
        return chunks

    merged: list[ImagePart | TextPart] = [chunks[0]]
    for chunk in chunks[1:]:
        if chunk["type"] == "text" and merged[-1]["type"] == "text":
            merged[-1] = TextPart(type="text", text=merged[-1]["text"] + chunk["text"])
        else:
            merged.append(chunk)
    return merged


class Qwen3Renderer(Renderer):
    """Renderer for Qwen3 models with thinking enabled.

    Format:
        <|im_start|>system
        You are Qwen, created by Alibaba Cloud.<|im_end|>
        <|im_start|>user
        What can you help me with?<|im_end|>
        <|im_start|>assistant
        <think>
        [reasoning content]
        </think>
        I can help you with...<|im_end|>
    """

    def __init__(self, tokenizer, strip_thinking_from_history: bool = True):
        super().__init__(tokenizer)
        self.strip_thinking_from_history = strip_thinking_from_history

    @property
    def has_extension_property(self) -> bool:
        return not self.strip_thinking_from_history

    def _get_qwen_role_for_message(self, message: Message) -> str:
        role = message["role"]
        if role == "tool":
            return "user"
        return role

    def _wrap_qwen_tool_response(self, content: str) -> str:
        return f"<tool_response>\n{content}\n</tool_response>"

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        maybe_newline = "\n" if ctx.idx > 0 else ""

        role = self._get_qwen_role_for_message(message)
        header_str = f"{maybe_newline}<|im_start|>{role}\n"

        content = message["content"]

        if isinstance(content, list):
            parts = content
            if self.strip_thinking_from_history and message["role"] == "assistant" and not ctx.is_last:
                parts = remove_thinking(parts)
            rendered_parts = []
            for p in parts:
                if p["type"] == "thinking":
                    rendered_parts.append(f"<think>{p['thinking']}</think>")
                elif p["type"] == "text":
                    rendered_parts.append(p["text"])
            output_content = "".join(rendered_parts)
        else:
            output_content = content

        if message["role"] == "tool":
            output_content = self._wrap_qwen_tool_response(output_content)

        if "tool_calls" in message:
            output_content += "\n" + "\n".join(
                [
                    f"<tool_call>\n{json.dumps(_tool_call_payload(tool_call))}\n</tool_call>"
                    for tool_call in message["tool_calls"]
                ]
            )
        output_content += "<|im_end|>"

        header = EncodedTextChunk(tokens=self.tokenizer.encode(header_str, add_special_tokens=False))
        output: list[ModelInputChunk] = [
            EncodedTextChunk(tokens=self.tokenizer.encode(output_content, add_special_tokens=False))
        ]
        return RenderedMessage(header=header, output=output)

    @property
    def _end_message_token(self) -> int:
        tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        assert len(tokens) == 1, f"Expected single token for <|im_end|>, got {len(tokens)}"
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        assert isinstance(assistant_message["content"], str)
        content = assistant_message["content"]

        parts = parse_content_blocks(content)

        if parts is not None:
            assistant_message["content"] = parts

            tool_calls = [p["tool_call"] for p in parts if p["type"] == "tool_call"]
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls

            unparsed = [
                UnparsedToolCall(raw_text=p["raw_text"], error=p["error"])
                for p in parts
                if p["type"] == "unparsed_tool_call"
            ]
            if unparsed:
                assistant_message["unparsed_tool_calls"] = unparsed
        else:
            assistant_message["content"] = content

        return assistant_message, True


class Qwen3VLRenderer(Qwen3Renderer):
    """Vision-language renderer for Qwen3-VL models with thinking support."""

    image_processor: ImageProcessor

    def __init__(
        self,
        tokenizer,
        image_processor: ImageProcessor,
        strip_thinking_from_history: bool = True,
        merge_text_chunks: bool = True,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.strip_thinking_from_history = strip_thinking_from_history
        self.merge_text_chunks = merge_text_chunks

    def _preprocess_message_parts(
        self, message: Message, *, strip_thinking: bool = False
    ) -> list[ImagePart | TextPart]:
        """Convert message content to list form for VL rendering.

        Wraps images with <|vision_start|> / <|vision_end|> tokens.
        """
        content = message["content"]
        if isinstance(content, str):
            base_parts: list[ImagePart | TextPart] = [TextPart(type="text", text=content)]
        else:
            base_parts = []
            for p in content:
                if p["type"] == "text":
                    base_parts.append(cast(TextPart, p))
                elif p["type"] == "image":
                    base_parts.append(cast(ImagePart, p))
                elif p["type"] == "thinking":
                    if not strip_thinking:
                        base_parts.append(TextPart(type="text", text=f"<think>{p['thinking']}</think>"))

        # Wrap images with vision tokens
        chunks: list[ImagePart | TextPart] = []
        for content_chunk in base_parts:
            if content_chunk["type"] == "image":
                chunks.append(TextPart(type="text", text="<|vision_start|>"))
            chunks.append(content_chunk)
            if content_chunk["type"] == "image":
                chunks.append(TextPart(type="text", text="<|vision_end|>"))

        return chunks

    def _wrap_qwen_tool_response_chunks(
        self, chunks: list[ImagePart | TextPart]
    ) -> list[ImagePart | TextPart]:
        return (
            [TextPart(type="text", text="<tool_response>\n")]
            + chunks
            + [TextPart(type="text", text="\n</tool_response>")]
        )

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        maybe_newline = "\n" if ctx.idx > 0 else ""

        role = self._get_qwen_role_for_message(message)
        header_str = f"{maybe_newline}<|im_start|>{role}\n"

        strip_thinking = (
            self.strip_thinking_from_history and message["role"] == "assistant" and not ctx.is_last
        )
        output_chunks = self._preprocess_message_parts(message, strip_thinking=strip_thinking)

        if message["role"] == "tool":
            output_chunks = self._wrap_qwen_tool_response_chunks(output_chunks)

        if "tool_calls" in message:
            output_chunks += [
                TextPart(
                    type="text",
                    text="\n"
                    + "\n".join(
                        [
                            f"<tool_call>\n{json.dumps(_tool_call_payload(tool_call))}\n</tool_call>"
                            for tool_call in message["tool_calls"]
                        ]
                    ),
                )
            ]
        output_chunks += [TextPart(type="text", text="<|im_end|>")]

        if self.merge_text_chunks:
            output_chunks = _merge_consecutive_text_parts(output_chunks)

        output_chunks_encoded: list[ModelInputChunk] = [
            image_to_chunk(
                image_or_str=x["image"],
                image_processor=cast(ImageProcessorProtocol, self.image_processor),
            )
            if x["type"] == "image"
            else EncodedTextChunk(tokens=self.tokenizer.encode(x["text"], add_special_tokens=False))
            for x in output_chunks
        ]

        header = EncodedTextChunk(tokens=self.tokenizer.encode(header_str, add_special_tokens=False))
        return RenderedMessage(header=header, output=output_chunks_encoded)
