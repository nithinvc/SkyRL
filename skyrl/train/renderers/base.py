"""
Base types, utilities, and abstract Renderer class for message rendering.

Trimmed from tinker-cookbook/tinker_cookbook/renderers/base.py.
Streaming types, Utf8TokenDecoder, build_supervised_example, and TrainOnWhat are omitted.
"""

import io
import json
import logging
import re
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, Optional, Protocol, TypedDict, Union

import pydantic
from PIL import Image

from skyrl.tinker.types import EncodedTextChunk, ImageChunk, ModelInput, ModelInputChunk

logger = logging.getLogger(__name__)

# Type alias for tokenizer — avoids slow import of PreTrainedTokenizer at module level
Tokenizer = Any


# ---------------------------------------------------------------------------
# Tool types (based on kosong)
# ---------------------------------------------------------------------------


class StrictBase(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    def __str__(self) -> str:
        return repr(self)


class ToolCall(StrictBase):
    """Structured tool invocation following OpenAI/kosong format."""

    class FunctionBody(pydantic.BaseModel):
        name: str
        arguments: str  # JSON string

    type: Literal["function"] = "function"
    id: str | None = None
    function: FunctionBody


class UnparsedToolCall(StrictBase):
    """Tool call that failed to parse from model output."""

    raw_text: str
    error: str


# ---------------------------------------------------------------------------
# Content part types
# ---------------------------------------------------------------------------


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    type: Literal["image"]
    image: str | Image.Image


class ThinkingPart(TypedDict):
    type: Literal["thinking"]
    thinking: str


class ToolCallPart(TypedDict):
    type: Literal["tool_call"]
    tool_call: ToolCall


class UnparsedToolCallPart(TypedDict):
    type: Literal["unparsed_tool_call"]
    raw_text: str
    error: str


ContentPart = TextPart | ImagePart | ThinkingPart | ToolCallPart | UnparsedToolCallPart

Role = str
Content = str | list[ContentPart]


class Message(TypedDict):
    """Container for a single turn in a multi-turn conversation."""

    role: Role
    content: Content
    tool_calls: NotRequired[list[ToolCall]]
    unparsed_tool_calls: NotRequired[list[UnparsedToolCall]]
    trainable: NotRequired[bool]
    tool_call_id: NotRequired[str]
    name: NotRequired[str]


# ---------------------------------------------------------------------------
# Render context and rendered message
# ---------------------------------------------------------------------------


@dataclass
class RenderContext:
    """Context passed to render_message for rendering a single message."""

    idx: int
    is_last: bool
    prev_message: Message | None = None


@dataclass(frozen=True)
class RenderedMessage:
    """Container for parts of a rendered message, structured for loss masking.

    output: what the model generates for this turn (trainable tokens).
    header: role identifier/delimiters (non-trainable).
    stop_overlap: tokens overlapping stop sequence and next header (rare).
    """

    output: list[ModelInputChunk]
    header: EncodedTextChunk | None = None
    stop_overlap: EncodedTextChunk | None = None


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------


def ensure_text(content: Content) -> str:
    if isinstance(content, str):
        return content
    if len(content) == 1 and content[0]["type"] == "text":
        return content[0]["text"]
    raise ValueError(f"Expected text content, got multimodal content with {len(content)} parts")


def ensure_list(content: Content) -> list[ContentPart]:
    if isinstance(content, str):
        return [TextPart(type="text", text=content)]
    return content


def remove_thinking(parts: list[ContentPart]) -> list[ContentPart]:
    return [p for p in parts if p["type"] != "thinking"]


def get_text_content(message: Message) -> str:
    """Extract text content from message, stripping thinking parts."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content if p["type"] == "text")


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------


def _parse_tool_call_json(tool_call_str: str, raw_text: str) -> ToolCall | UnparsedToolCall:
    try:
        tool_call = json.loads(tool_call_str.strip())
    except json.JSONDecodeError as e:
        return UnparsedToolCall(raw_text=raw_text, error=f"Invalid JSON: {e}")

    if not isinstance(tool_call, dict):
        return UnparsedToolCall(raw_text=raw_text, error="Tool call is not a JSON object")

    name = tool_call.get("name")
    arguments = tool_call.get("arguments")
    tool_id = tool_call.get("id")

    if not isinstance(name, str):
        return UnparsedToolCall(raw_text=raw_text, error="Missing or invalid 'name' field")
    if not isinstance(arguments, dict):
        return UnparsedToolCall(raw_text=raw_text, error="Missing or invalid 'arguments' field")

    if tool_id is not None and not isinstance(tool_id, str):
        tool_id = None

    return ToolCall(
        function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments)),
        id=tool_id,
    )


def _tool_call_payload(tool_call: ToolCall) -> dict[str, object]:
    """Minimal JSON payload for embedding in <tool_call> blocks."""
    return {
        "name": tool_call.function.name,
        "arguments": json.loads(tool_call.function.arguments),
    }


def parse_content_blocks(content: str) -> list[ContentPart] | None:
    """Parse a string with <think>...</think> and <tool_call>...</tool_call> tags.

    Returns None if no special tags found.
    """
    if "<think>" not in content and "<tool_call>" not in content:
        return None

    parts: list[ContentPart] = []
    pos = 0
    pattern = re.compile(r"<think>(.*?)</think>|<tool_call>(.*?)</tool_call>", re.DOTALL)

    for match in pattern.finditer(content):
        text_before = content[pos : match.start()]
        if text_before:
            parts.append(TextPart(type="text", text=text_before))

        if match.group(1) is not None:
            thinking = match.group(1)
            if thinking:
                parts.append(ThinkingPart(type="thinking", thinking=thinking))
        else:
            tool_call_json = match.group(2)
            raw_text = match.group(0)
            parsed = _parse_tool_call_json(tool_call_json, raw_text)
            if isinstance(parsed, UnparsedToolCall):
                parts.append(
                    UnparsedToolCallPart(type="unparsed_tool_call", raw_text=parsed.raw_text, error=parsed.error)
                )
            else:
                parts.append(ToolCallPart(type="tool_call", tool_call=parsed))

        pos = match.end()

    remaining = content[pos:]
    if remaining:
        parts.append(TextPart(type="text", text=remaining))

    return parts


def parse_response_for_stop_token(
    response: list[int], tokenizer: Tokenizer, stop_token: int
) -> tuple[Message, bool]:
    """Parse a response that should end with a stop token.

    Returns (Message, success). If the stop token is missing, success is False.
    """
    emt_count = response.count(stop_token)
    if emt_count == 0:
        str_response = tokenizer.decode(response)
        logger.debug(f"Response is not a valid assistant response: {str_response}")
        return Message(role="assistant", content=str_response), False
    elif emt_count == 1:
        str_response = tokenizer.decode(response[: response.index(stop_token)])
        return Message(role="assistant", content=str_response), True
    else:
        raise ValueError(
            f"When parsing response, expected to split into 1 or 2 pieces using stop tokens, but got {emt_count}. "
            "You probably are using the wrong stop tokens when sampling"
        )


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------


class ImageProcessorProtocol(Protocol):
    merge_size: int
    patch_size: int

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: Optional[dict] = None
    ) -> int:
        raise NotImplementedError()


def image_to_chunk(
    image_or_str: Image.Image | str, image_processor: ImageProcessorProtocol
) -> ImageChunk:
    """Convert a PIL Image (or URL) to an ImageChunk for VL models."""
    if isinstance(image_or_str, str):
        with urllib.request.urlopen(image_or_str) as response:
            pil_image = Image.open(io.BytesIO(response.read()))
    elif isinstance(image_or_str, Image.Image):
        pil_image = image_or_str
    else:
        raise ValueError("The provided image must be a PIL.Image.Image, URL, or data URI.")

    if pil_image.mode in ("RGBA", "LA", "P"):
        pil_image = pil_image.convert("RGB")

    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="JPEG")
    image_data = img_byte_arr.getvalue()

    width, height = pil_image.size
    num_image_tokens = (
        image_processor.get_number_of_image_patches(height, width, images_kwargs={}) // image_processor.merge_size**2
    )

    return ImageChunk(
        data=image_data,
        format="jpeg",
        expected_tokens=num_image_tokens,
    )


# ---------------------------------------------------------------------------
# Renderer ABC
# ---------------------------------------------------------------------------


class Renderer(ABC):
    """Abstract base class for rendering message lists into sampling prompts.

    Subclasses must implement:
    - get_stop_sequences(): Return stop tokens for sampling
    - render_message(): Break a message into header/output components
    - parse_response(): Convert sampled tokens back into a Message
    """

    tokenizer: Tokenizer

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def has_extension_property(self) -> bool:
        """Whether this renderer satisfies the sequence extension property."""
        return False

    @property
    def _bos_tokens(self) -> list[int]:
        return []

    @abstractmethod
    def get_stop_sequences(self) -> list[str] | list[int]:
        ...

    @abstractmethod
    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        ...

    @abstractmethod
    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        ...

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        """Return tokens to append to the prompt for generation (role header)."""
        rendered = self.render_message(Message(role=role, content=""), ctx)
        if rendered.header:
            return list(rendered.header.tokens)
        return []

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> ModelInput:
        """Build a ModelInput (chunks) for sampling from the model.

        Args:
            messages: Conversation history.
            role: Role of the partial message to be completed.
            prefill: Optional string to prefill in the model's generation.
        """
        chunks: list[ModelInputChunk] = []
        if self._bos_tokens:
            chunks.append(EncodedTextChunk(tokens=self._bos_tokens))
        for idx, message in enumerate(messages):
            ctx = RenderContext(
                idx=idx,
                is_last=(idx == len(messages) - 1),
                prev_message=messages[idx - 1] if idx > 0 else None,
            )
            rendered_message = self.render_message(message, ctx)
            if rendered_message.header:
                chunks.append(rendered_message.header)
            chunks.extend([x for x in rendered_message.output if x])

        suffix_ctx = RenderContext(
            idx=len(messages),
            is_last=True,
            prev_message=messages[-1] if messages else None,
        )
        suffix_tokens = self._get_generation_suffix(role, suffix_ctx)
        if suffix_tokens:
            chunks.append(EncodedTextChunk(tokens=suffix_tokens))

        if prefill:
            chunks.append(EncodedTextChunk(tokens=self.tokenizer.encode(prefill, add_special_tokens=False)))

        return ModelInput(chunks=chunks)
