from __future__ import annotations

import asyncio
import base64
import logging
from typing import TYPE_CHECKING, Union

from skyrl.tinker.types import (
    EncodedTextChunk,
    ImageAssetPointerChunk,
    ImageChunk,
    ModelInput,
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


class VLLMRenderer:
    """Renders ModelInputs by calling vLLM's /v1/chat/completions/render for image placeholders."""

    def __init__(self, client: RemoteInferenceClient, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    def __call__(self, model_inputs: list[ModelInput]) -> list[RenderedModelInput]:
        """Synchronous entry point. Renders all ModelInputs, resolving image placeholders via vLLM."""

        async def _render_all():
            tasks = []
            for mi in model_inputs:
                tasks.append(self._render_model_input(mi))
            return await asyncio.gather(*tasks)

        return asyncio.run(_render_all())

    async def _render_model_input(self, model_input: ModelInput) -> RenderedModelInput:
        # Collect image chunks with their indices
        image_chunks: list[Union[ImageChunk, ImageAssetPointerChunk]] = []
        for chunk in model_input.chunks:
            if isinstance(chunk, (ImageChunk, ImageAssetPointerChunk)):
                image_chunks.append(chunk)

        # Fast path: text only — no HTTP call needed
        if not image_chunks:
            prompt_ids: list[int] = []
            for chunk in model_input.chunks:
                if isinstance(chunk, EncodedTextChunk):
                    prompt_ids.extend(chunk.tokens)
            return RenderedModelInput(prompt_ids=prompt_ids)

        # Render images via vLLM
        placeholder_tokens, placeholders, mm_kwargs = await self._render_images(image_chunks)

        # Assemble prompt_ids and update placeholder offsets by walking chunks in order
        prompt_ids = []
        mm_placeholders: list[MultiModalPlaceholder] = []
        image_idx = 0
        for chunk in model_input.chunks:
            if isinstance(chunk, EncodedTextChunk):
                prompt_ids.extend(chunk.tokens)
            elif isinstance(chunk, (ImageChunk, ImageAssetPointerChunk)):
                placeholder = placeholders[image_idx]
                placeholder.offset = len(prompt_ids)
                prompt_ids.extend(placeholder_tokens[image_idx])
                mm_placeholders.append(placeholder)
                image_idx += 1

        return RenderedModelInput(
            prompt_ids=prompt_ids,
            multi_modal_placeholders=mm_placeholders if mm_placeholders else None,
            multi_modal_kwargs=mm_kwargs if mm_kwargs else None,
        )

    async def _render_images(
        self,
        image_chunks: list[Union[ImageChunk, ImageAssetPointerChunk]],
    ) -> tuple[list[list[int]], list[MultiModalPlaceholder], dict[str, bytes]]:
        """Render all images in a single /v1/chat/completions/render call.

        Returns:
            - placeholder tokens per image (spliced from the render response)
            - MultiModalPlaceholder stubs (offset=0, adjusted by caller)
            - multi_modal_kwargs (Empty till upstreamed in vLLM)
        """
        content_parts = []
        for chunk in image_chunks:
            if isinstance(chunk, ImageChunk):
                # TODO (nithinc): see if there is a more efficient way - maybe we just pass the base64 string?
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
        # mm_placeholders: dict[modality str, PlaceholderInfo]
        image_placeholders = features.get("mm_placeholders", {}).get("image", [])

        if len(image_placeholders) != len(image_chunks):
            raise RuntimeError(f"Expected {len(image_chunks)} image placeholders, got {len(image_placeholders)}")

        placeholder_tokens: list[list[int]] = []
        placeholders: list[MultiModalPlaceholder] = []
        for i, placeholder in enumerate(image_placeholders):
            offset = placeholder["offset"]
            length = placeholder["length"]
            tokens = token_ids[offset : offset + length]
            placeholder_tokens.append(tokens)
            placeholders.append(MultiModalPlaceholder(offset=0, length=length))

            chunk = image_chunks[i]
            if chunk.expected_tokens is not None and chunk.expected_tokens != length:
                logger.warning(
                    f"Image {i}: expected_tokens={chunk.expected_tokens} but render returned {length} placeholder tokens"
                )

        return placeholder_tokens, placeholders, {}
