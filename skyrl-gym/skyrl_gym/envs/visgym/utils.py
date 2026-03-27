import base64
import io
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image


def encode_image_to_base64(rgb_array: np.ndarray, format: str = "png") -> str:
    """Convert an RGB numpy array to a base64-encoded string."""
    img = Image.fromarray(rgb_array)
    buffer = io.BytesIO()
    img.save(buffer, format=format.upper())
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def make_image_message(text: str, rgb_array: np.ndarray, role: str = "user") -> Dict[str, Any]:
    """Build an OpenAI-format multimodal message with text and an image.

    Returns a dict like:
        {"role": "user", "content": [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]}
    """
    b64 = encode_image_to_base64(rgb_array)
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]
    return {"role": role, "content": content}


# Regex to find the first Python tuple pattern like ('move', 0) or ('stop', 'stop')
# Handles nested tuples/lists for actions like ('swap', ((0,0),(1,1))) and ('reorder', [0,1,2,3])
_ACTION_TUPLE_RE = re.compile(
    r"""\(\s*['"](\w+)['"]\s*,\s*"""  # opening: ('branch_name',
    r"""(.+?)"""                        # payload (non-greedy)
    r"""\)""",                          # closing )
    re.DOTALL,
)


def extract_action(vlm_output: str) -> tuple[str, bool]:
    """Extract the first action tuple from VLM output text.

    Searches for a pattern like ('move', 0) or ('stop', 'stop') in the VLM's
    full text output (which may contain reasoning before/after the action).

    Returns:
        (action_string, matched): The extracted tuple string and whether a regex
        match was found. If no match, returns (stripped raw input, False) so
        VisGym's own ast.literal_eval handles the error gracefully.
    """
    match = _ACTION_TUPLE_RE.search(vlm_output)
    if match:
        return match.group(0), True
    return vlm_output.strip(), False
