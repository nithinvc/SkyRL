# Vision Token Misalignment Bug — Phase 2 Progress

**Status: ALL BUGS FIXED. Training runs end-to-end successfully.**

## Starting Point

Phase 1 (`progress.md`) identified the root cause of the `ValueError: Vision token misalignment` crash in `convert_to_training_input`. Phase 2 implemented fixes for that and four additional downstream issues uncovered during iterative debugging.

---

## Bug 1: Vision Token Misalignment (FIXED)

### Root Cause

Two interleaved issues in `trainer.py:convert_to_training_input`:

1. **Ordering mismatch** between generator and renderer:

| Component | Token order per turn |
|-----------|---------------------|
| Generator `response_ids` | `[model_response, suffix_end_zeros, obs_zeros]` |
| VLLMRenderer | `[suffix, model_response, end_token, obs_tokens]` |

2. **Image placeholder count differences** between the local `image_processor` (`expected_tokens`) and vLLM's actual rendered counts.

The original code naively split the rendered sequence using `rendered[:-resp_len]`, applying `loss_mask` to wrong tokens and breaking when placeholder counts differed.

### Fix

**File: `skyrl/train/trainer.py`**

- New function `_rebuild_vision_response()` recomputes the prompt/response split by walking `model_input.chunks` with actual vLLM placeholder lengths, then rebuilds `loss_mask`, `rewards`, and `logprobs` per-turn in rendered order.
- `render_vision_inputs()` now returns `mm_placeholders_list` (4-tuple).
- `convert_to_training_input()` calls `_rebuild_vision_response()` per sample.

---

## Bug 2: RefWorkerBase KeyError on pixel_values (FIXED)

### Root Cause

`RefWorkerBase._forward_micro_batch()` used unsafe `micro_batch["pixel_values"]` while `PolicyWorkerBase` correctly used `.get()`.

### Fix

**File: `skyrl/backends/skyrl_train/workers/worker.py`**, line 1229-1230: Changed to `.get("pixel_values", None)` and `.get("image_grid_thw", None)`.

---

## Bug 3: Sample Packing + VLM Assertion (FIXED)

### Root Cause

Config defaults to `use_sample_packing: True` (`config.py:597`). VLMs require it off because they use model-specific 3D positional IDs incompatible with Flash Attention 2's variable-length packing.

### Fix

**File: `examples/train/visgym/run_visgym.sh`**: Added `trainer.use_sample_packing=false`.

---

## Bug 4: pixel_values Not Propagated from vLLM (FIXED)

### Root Cause

The vLLM render endpoint (`/v1/chat/completions/render`) returns `kwargs_data: {"image": [None, None, ...]}` — the list structure exists but every item is `None`. This is **by design**: vLLM's multimodal processor LRU cache strips tensor data on cache hits, returning `None` to indicate "resolve from cache". This works for the generate endpoint (which holds cached tensors), but the training path needs actual tensor data.

The `render_vision_inputs` method checked `has_vision = any(r.multi_modal_kwargs for r in rendered)`, which was always `False` since all kwargs_data items were `None`, so `pixel_values` was never populated.

### Fix

**File: `skyrl/train/trainer.py`**: Added `_compute_pixel_values_locally()` method as a fallback. When `kwargs_data` is unavailable, it loads `AutoImageProcessor` from the policy model path and processes raw JPEG images from `ImageChunk.data` into `pixel_values` and `image_grid_thw` tensors locally. The processor is cached after first load.

---

## Bug 5: FSDP CPU Offload Device Mismatch (FIXED)

### Root Cause

The ref model was configured with `cpu_offload=true`. FSDP CPU offloading left the rotary embedding's `inv_freq` buffer on CPU, while `position_ids` (computed from GPU-resident `attention_mask`) was on CUDA. The matrix multiply `inv_freq @ position_ids` in Qwen3-VL's rotary embedding crashed with `RuntimeError: Expected all tensors to be on the same device`.

### Fix

**File: `examples/train/visgym/run_visgym.sh`**: Changed `trainer.ref.fsdp_config.cpu_offload=false`. Qwen3-VL-2B is small enough to fit in GPU memory without offloading.

---

## Files Modified

| File | Change |
|------|--------|
| `skyrl/train/trainer.py` | Added `_rebuild_vision_response()`, `_compute_pixel_values_locally()`, modified `render_vision_inputs()` and `convert_to_training_input()` |
| `skyrl/backends/skyrl_train/workers/worker.py` | Safe `.get()` access for pixel_values/image_grid_thw in RefWorkerBase |
| `examples/train/visgym/run_visgym.sh` | Added `use_sample_packing=false`, changed `cpu_offload=false` |

## Verification

Training completed end-to-end: 3 epochs × 4 batches = 12 training steps. All steps completed with metrics logged. First-run metrics show `final_loss: 0.0` and `avg_raw_reward: 0.0` — expected since the untrained model hasn't learned the maze task yet.

## Architecture Notes

- The generator builds `response_ids` in **generator order** (model output first, then template/observation tokens as zeros)
- The VLLMRenderer produces tokens in **conversation order** (template tokens first, then model output, then end token, then observations)
- `_rebuild_vision_response()` bridges these orderings via the `model_input.chunks` structure
- vLLM's `kwargs_data` from the render endpoint cannot be relied on for pixel_values in the training path (LRU cache strips tensor data). The local `AutoImageProcessor` fallback is the correct approach.
- The suffix (assistant header tokens like `\n<|im_start|>assistant\n`) has constant length across all turns (4 tokens for Qwen3-VL)
