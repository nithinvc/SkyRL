# Vision Language Inputs in Tinker (Summary)

This report summarizes how Tinker handles vision‑language inputs, the renderer abstraction, and how `Datum` is used in training, based on the cookbook source.

## Vision‑Language Inputs

Core idea: **all model inputs are `ModelInput`, which is an ordered list of chunks**. For multimodal, the list can mix text and images:

- `EncodedTextChunk`: token IDs for text
- `ImageChunk`: raw image bytes + metadata (`format`, optional `expected_tokens`)

There are two levels of usage:

1. **Low‑level (explicit chunks)**
   - You manually build a `ModelInput(chunks=[...])` with text + `ImageChunk`.
   - For Qwen3‑VL, you must include special vision tokens like `<|vision_start|>` / `<|vision_end|>` around the image unless you use a renderer.
   - Image assets are capped at 2MB in the cookbook examples.

2. **High‑level (renderer abstraction)**
   - You pass structured messages with `ImagePart` + `TextPart`.
   - The renderer handles image preprocessing, special tokens, and tokenization.

Relevant docs/files:
- `docs/training-sampling.mdx` (low‑level `ImageChunk` usage)
- `docs/rendering.mdx` (high‑level multimodal messages)
- `docs/api-reference/types.md` (`ImageChunk` fields)

## Renderer Class (What It Is)

A renderer is a **conversation compiler** that converts structured messages into model‑ready inputs and training targets. It is more than a Hugging Face processor because it also encodes training masks and parsing logic.

Key responsibilities (from `tinker_cookbook/renderers/base.py`):

- `build_generation_prompt(messages, ...) -> ModelInput`
  - Produces the `ModelInput` used for sampling/inference.
- `build_supervised_example(messages, ...) -> (ModelInput, weights)`
  - Produces token‑level weights aligned with the token sequence.
  - These weights control which tokens receive loss.
- `get_stop_sequences()`
  - Defines stop tokens/strings for sampling.
- `parse_response(tokens)`
  - Converts sampled tokens back into a structured message.

### Multimodal message interface

Messages can include content parts:

- `TextPart(type="text", text=...)`
- `ImagePart(type="image", image=...)` (image can be URL or PIL image)

Example usage (from `docs/rendering.mdx`):

```python
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.image_processing_utils import get_image_processor

model_name = "Qwen/Qwen3-VL-235B-A22B-Instruct"
tokenizer = tokenizer_utils.get_tokenizer(model_name)
image_processor = get_image_processor(model_name)

renderer = renderers.Qwen3VLInstructRenderer(tokenizer, image_processor)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://.../tinker-cover.png"},
            {"type": "text", "text": "What is in this image?"},
        ],
    }
]

prompt = renderer.build_generation_prompt(messages)
```

### Extension property

Renderers may or may not satisfy the **extension property** (`renderer.has_extension_property`).

- Extension property means each successive observation is a **prefix extension** of the previous one.
- If it holds, RL training can merge multiple timesteps into one `Datum` and reuse KV‑cache, giving **O(T)** compute.
- If it breaks (e.g., stripping `<think>` blocks from history), RL must create multiple `Datum`s and compute scales **O(T²)**.

See: `docs/rl/sequence-extension.mdx` and `tinker_cookbook/rl/data_processing.py`.

## Datum (Training Unit)

A `Datum` is the **unit of training** sent to Tinker. It always includes:

- `model_input`: the `ModelInput` (tokens + image chunks)
- `loss_fn_inputs`: loss‑specific tensors (e.g., target tokens, logprobs, advantages, masks, weights)

### Where `Datum` is used in VLM example

In the VLM classifier recipe, `Datum` is created inside the dataset builder:

- `build_supervised_example(...)` returns `(ModelInput, weights)` using the renderer.
- `datum_from_model_input_weights(...)` wraps those into a `tinker.Datum`.
- `get_batch(...)` returns a list of `Datum`s for training.

Relevant code:
- `tinker_cookbook/recipes/vlm_classifier/data.py` (`build_supervised_example`, `get_batch`)
- `tinker_cookbook/supervised/common.py` (`datum_from_model_input_weights`)

## Relationship Summary

- **Renderer output**: `ModelInput` (+ optional token weights)
- **Datum**: `ModelInput` + loss inputs packaged for training
- **VLM inputs**: `ModelInput` with interleaved `EncodedTextChunk` and `ImageChunk`

This division keeps the **prompting/rendering concerns** (renderer) separate from **training mechanics** (Datum).
