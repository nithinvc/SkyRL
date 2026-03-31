# Vision Token Misalignment Bug — Progress Notes

## Problem

Running `bash examples/train/visgym/run_visgym.sh` crashes after trajectory generation completes, at the `convert_to_training_input` step:

```
ValueError: Vision token misalignment at response position 0: rendered=198, generator=492.
Image placeholder count mismatch between generator and VLLMRenderer.
```

## Root Cause Analysis

The bug is a **token count mismatch between two independent image processors**:

### Two paths compute image placeholder token counts independently:

1. **Generator path** (`skyrl_gym_tinker_generator.py`):
   - Uses local `image_processor.get_number_of_image_patches()` via `image_to_chunk()` in `renderers/base.py:285-317`
   - Stores result as `ImageChunk.expected_tokens`
   - `_flatten_all_tokens()` (line 73) uses `expected_tokens` to emit `[0] * expected_tokens` placeholders
   - This determines `len(prompt_token_ids)` and `len(response_ids)` — the split point

2. **Training path** (`backends/renderer.py` — `VLLMRenderer`):
   - Sends images to vLLM's `/v1/chat/completions/render` endpoint
   - vLLM returns its own placeholder token count per image
   - `_render_single()` uses vLLM's count (not `expected_tokens`)
   - This determines `len(rendered_prompt_ids)` — the actual rendered sequence length

### How the crash happens:

In `trainer.py:convert_to_training_input` (line 693-712):
```python
for rendered, resp, lm in zip(rendered_prompt_ids, response_ids, loss_masks):
    resp_len = len(resp)
    prompt_ids.append(rendered[:-resp_len] if resp_len > 0 else rendered)
    replaced = rendered[-resp_len:] if resp_len > 0 else []
```

- `rendered` length = text tokens + **vLLM's** image placeholder counts
- `resp_len` = `len(response_ids)` computed using **local** `expected_tokens`
- If `expected_tokens != vLLM_placeholder_count` for ANY image, the split point is wrong
- The "response" tail overlaps into prompt tokens (or leaves a gap)
- First `loss_mask==1` position compares unrelated token IDs → ValueError

### The VLLMRenderer already warns about this (but doesn't fix it):

In `renderer.py:140-143`:
```python
if chunk.expected_tokens is not None and chunk.expected_tokens != length:
    logger.warning(
        f"Image {i}: expected_tokens={chunk.expected_tokens} but render returned {length} placeholder tokens"
    )
```

## Fix Strategy

The fix needs to ensure `prompt_token_ids + response_ids` length matches `rendered_prompt_ids` length. Two approaches:

### Approach A: Update `response_ids` accounting when VLLMRenderer returns different counts (Recommended)

In `convert_to_training_input`, instead of naively splitting `rendered[:-resp_len]`, recompute the split by comparing chunk-level token counts from the `model_input` against the rendered output. Essentially, recalculate the observation token placeholders using the vLLM counts instead of `expected_tokens`.

### Approach B: Make the generator use vLLM's placeholder count from the start

Have the generator call vLLM's render endpoint to get the true placeholder count when building `ImageChunk`s, so `expected_tokens` always matches. This is more invasive and adds latency to generation.

### Approach C (Simplest): Recompute the prompt/response boundary using the model_input chunks

Since `convert_to_training_input` has access to `model_input` (which contains `ImageChunk`s with `expected_tokens`), and VLLMRenderer returns the actual rendered count per image, we can compute the **delta** between expected and actual tokens across all images, and adjust the split point accordingly.

Actually the simplest correct fix: in `convert_to_training_input`, when `vision_data` is available, don't split based on `len(response_ids)`. Instead, reconstruct which tokens are prompt vs response by re-walking the `model_input` chunks and comparing against `rendered_prompt_ids`, accounting for the actual vLLM placeholder lengths. Then update `response_ids` to match.

## Key Files

| File | Role |
|------|------|
| `skyrl/train/trainer.py:666-785` | `convert_to_training_input` — where the crash occurs |
| `skyrl/train/trainer.py:612-664` | `render_vision_inputs` — calls VLLMRenderer |
| `skyrl/backends/renderer.py:35-151` | `VLLMRenderer` — renders via vLLM, uses vLLM's placeholder count |
| `skyrl/train/generators/skyrl_gym_tinker_generator.py:53-81` | `_flatten_all_tokens` / `_count_chunk_tokens` — uses `expected_tokens` |
| `skyrl/train/generators/skyrl_gym_tinker_generator.py:121-306` | `agent_loop` — builds `response_ids` with `expected_tokens`-based accounting |
| `skyrl/train/renderers/base.py:285-317` | `image_to_chunk` — computes `expected_tokens` via local `image_processor` |
| `examples/train/visgym/run_visgym.sh` | Launch script |
| `examples/train/visgym/visgym_entrypoint.py` | Entry point |

## How observation images cause the mismatch

In `agent_loop` lines 258-281, when the environment returns image observations:
- `render_message(obs_msg)` produces `ImageChunk`s with `expected_tokens` from local processor
- These are appended to `running_chunks` and their `expected_tokens` count is added to `all_response_ids` as `[0] * obs_token_count`
- But when `VLLMRenderer` later renders the full `final_model_input`, it gets **different** placeholder counts from vLLM for those same images
- Every image where `expected_tokens != vLLM_count` adds to the cumulative offset error

## Status

- [x] Identified the root cause
- [ ] Implement the fix
- [ ] Re-run training and verify it starts
