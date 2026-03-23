from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Union

from skyrl.tinker.types import (
    EncodedTextChunk,
    ImageAssetPointerChunk,
    ImageChunk,
    ModelInput,
    ModelInputChunk,
    MultiModalPlaceholder,
    RenderedModelInput,
)

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )

logger = logging.getLogger(__name__)


def render_model_input(model_inputs: list[ModelInput]) -> list[RenderedModelInput]:
    return [
        RenderedModelInput(
            prompt_ids=[tok for chunk in mi.chunks for tok in (chunk.tokens if hasattr(chunk, "tokens") else [])]
        )
        for mi in model_inputs
    ]


# ---------------------------------------------------------------------------
# Intermediate dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RenderedImage:
    """Per-image result extracted from vLLM render response."""

    placeholder_tokens: list[int]
    mm_hash: str | None = None


@dataclass
class VLLMPrompt:
    """vLLM-native prompt with multi-modal metadata"""

    token_ids: list[int]
    mm_hashes: dict[str, list[str]] | None = None
    mm_placeholders: dict[str, list[dict[str, int]]] | None = None


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def assemble_vllm_prompt(
    chunks: list[ModelInputChunk],
    rendered_images: list[RenderedImage],
) -> VLLMPrompt:
    """Walk ModelInputChunks in order, splicing in rendered image tokens.

    Builds ``token_ids``, ``mm_placeholders``, and ``mm_hashes``.
    """
    token_ids: list[int] = []
    placeholders: list[dict[str, int]] = []
    hashes: list[str] = []
    all_have_hash = True

    image_idx = 0
    for chunk in chunks:
        if isinstance(chunk, EncodedTextChunk):
            token_ids.extend(chunk.tokens)
        elif isinstance(chunk, (ImageChunk, ImageAssetPointerChunk)):
            ri = rendered_images[image_idx]
            offset = len(token_ids)
            token_ids.extend(ri.placeholder_tokens)
            placeholders.append({"offset": offset, "length": len(ri.placeholder_tokens)})
            if ri.mm_hash is not None:
                hashes.append(ri.mm_hash)
            else:
                all_have_hash = False
            image_idx += 1

    mm_placeholders: dict[str, list[dict[str, int]]] | None = {"image": placeholders} if placeholders else None
    mm_hashes: dict[str, list[str]] | None = None
    if placeholders and not all_have_hash:
        raise ValueError("All images must have a multi-modal hash.")
    if hashes:
        mm_hashes = {"image": hashes}

    return VLLMPrompt(token_ids=token_ids, mm_hashes=mm_hashes, mm_placeholders=mm_placeholders)


def vllm_prompt_to_rendered_model_input(prompt: VLLMPrompt) -> RenderedModelInput:
    """Convert an assembled VLLM prompt to the training-path RenderedModelInput."""
    mm_placeholders: list[MultiModalPlaceholder] | None = None
    if prompt.mm_placeholders and "image" in prompt.mm_placeholders:
        mm_placeholders = [
            MultiModalPlaceholder(offset=p["offset"], length=p["length"]) for p in prompt.mm_placeholders["image"]
        ]
    return RenderedModelInput(
        prompt_ids=prompt.token_ids,
        multi_modal_placeholders=mm_placeholders if mm_placeholders else None,
        multi_modal_kwargs=None,
    )


# ---------------------------------------------------------------------------
# VLLMRenderer
# ---------------------------------------------------------------------------


class VLLMRenderer:
    """Renders ModelInputs by calling vLLM's /v1/chat/completions/render for image placeholders."""

    def __init__(self, client: RemoteInferenceClient, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    def render_model_input(self, model_inputs: list[ModelInput]) -> list[RenderedModelInput]:
        """Training renderer. Renders all ModelInputs, resolving image placeholders via vLLM."""

        async def _render_all():
            return await asyncio.gather(*[self._render_single_model_input(mi) for mi in model_inputs])

        return asyncio.run(_render_all())

    def render_sample_prompt(self, model_inputs: list[ModelInput]) -> list[dict]:
        """Sampling renderer. Returns VLLMPrompts with mm metadata as dicts to be passed to vLLM."""

        async def _render_all():
            return await asyncio.gather(*[self._render_single_sample_prompt(mi) for mi in model_inputs])

        sample_prompts = asyncio.run(_render_all())
        return [asdict(sp) for sp in sample_prompts]

    # -- internal -------------------------------------------------------------

    async def _render_single_model_input(self, model_input: ModelInput) -> RenderedModelInput:
        image_chunks = [c for c in model_input.chunks if isinstance(c, (ImageChunk, ImageAssetPointerChunk))]

        # Fast path: text only — no HTTP call needed
        if not image_chunks:
            prompt_ids: list[int] = []
            for chunk in model_input.chunks:
                if isinstance(chunk, EncodedTextChunk):
                    prompt_ids.extend(chunk.tokens)
            return RenderedModelInput(prompt_ids=prompt_ids)

        # Step 1: render images
        rendered_images = await self._render_images(image_chunks)
        # Step 2: assemble vLLM prompt
        vllm_prompt = assemble_vllm_prompt(model_input.chunks, rendered_images)
        # Step 3: convert to training format
        return vllm_prompt_to_rendered_model_input(vllm_prompt)

    async def _render_single_sample_prompt(self, model_input: ModelInput) -> VLLMPrompt:
        image_chunks = [c for c in model_input.chunks if isinstance(c, (ImageChunk, ImageAssetPointerChunk))]

        if not image_chunks:
            prompt_ids = [tok for c in model_input.chunks if isinstance(c, EncodedTextChunk) for tok in c.tokens]
            return VLLMPrompt(token_ids=prompt_ids)

        rendered_images = await self._render_images(image_chunks)
        return assemble_vllm_prompt(model_input.chunks, rendered_images)

    async def _render_images(
        self,
        image_chunks: list[Union[ImageChunk, ImageAssetPointerChunk]],
    ) -> list[RenderedImage]:
        """Render all images in a single /v1/chat/completions/render call.

        Returns a RenderedImage per image with placeholder tokens and mm_hash.
        """
        content_parts = []
        for chunk in image_chunks:
            if isinstance(chunk, ImageChunk):
                b64_data = base64.b64encode(chunk.data).decode("ascii")
                url = f"data:image/{chunk.format};base64,{b64_data}"
            else:  # ImageAssetPointerChunk
                url = chunk.location
            content_parts.append({"type": "image_url", "image_url": {"url": url}})

        payload = {
            "json": {
                "model": self._model_name,
                "messages": [{"role": "user", "content": content_parts}],
            }
        }

        response = await self._client.render_chat_completion(payload)

        token_ids = response["token_ids"]
        features = response.get("features", {})
        image_placeholders = features.get("mm_placeholders", {}).get("image", [])
        image_hashes = features.get("mm_hashes", {}).get("image", [])

        if len(image_placeholders) != len(image_chunks):
            raise RuntimeError(f"Expected {len(image_chunks)} image placeholders, got {len(image_placeholders)}")

        rendered_images: list[RenderedImage] = []
        for i, placeholder in enumerate(image_placeholders):
            offset = placeholder["offset"]
            length = placeholder["length"]
            tokens = token_ids[offset : offset + length]

            mm_hash = image_hashes[i] if i < len(image_hashes) else None

            chunk = image_chunks[i]
            if chunk.expected_tokens is not None and chunk.expected_tokens != length:
                logger.warning(
                    f"Image {i}: expected_tokens={chunk.expected_tokens} but render returned {length} placeholder tokens"
                )

            rendered_images.append(RenderedImage(placeholder_tokens=tokens, mm_hash=mm_hash))

        return rendered_images
