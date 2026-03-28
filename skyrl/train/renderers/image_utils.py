"""
Utilities for working with image processors.

Avoids importing AutoImageProcessor until runtime (slow import).
Copied from tinker-cookbook/tinker_cookbook/image_processing_utils.py.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, TypeAlias

from PIL import Image

if TYPE_CHECKING:
    from transformers.image_processing_utils import BaseImageProcessor

    ImageProcessor: TypeAlias = BaseImageProcessor
else:
    ImageProcessor: TypeAlias = Any


@cache
def get_image_processor(model_name: str) -> ImageProcessor:
    model_name = model_name.split(":")[0]

    from transformers.models.auto.image_processing_auto import AutoImageProcessor

    try:
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    except Exception:
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
    return processor


def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    """Resize an image so that its longest side is at most max_size pixels."""
    width, height = image.size
    if max(width, height) <= max_size:
        return image

    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
