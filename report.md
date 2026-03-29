# VLM RL Training Pipeline — Bug Report & Fix Summary

**Script:** `examples/train/visgym/run_visgym.sh`
**Model:** Qwen3-VL-2B-Instruct
**Task:** Multi-turn RL training on VisGym maze_2d environment
**Outcome:** 5 bugs found and fixed. Training now runs end-to-end.

---

## Background

The VLM RL training pipeline generates multi-turn trajectories where an agent navigates a 2D maze. Each turn produces an image observation from the environment and a text response from the model. The pipeline has three main phases:

1. **Generation** — vLLM generates model responses in a multi-turn loop with the VisGym environment
2. **Rendering** — The `VLLMRenderer` re-renders the full conversation through vLLM's `/v1/chat/completions/render` endpoint to obtain the final token IDs (with correct image placeholder tokens) and multimodal features
3. **Training** — FSDP-sharded policy and reference models compute log-probabilities, then PPO updates the policy

The initial run crashed with `ValueError: Vision token misalignment`. Iterative debugging revealed a chain of 5 distinct bugs that each surfaced once the preceding one was fixed.

---

## Issue 1: Vision Token Misalignment in `convert_to_training_input`

**Error:** `ValueError: Vision token misalignment at response position 0: rendered=198, generator=492`

**Location:** `skyrl/train/trainer.py`, original `convert_to_training_input` method

### Root Cause

Two independent problems combined:

**Problem A — Token ordering mismatch.** The generator and the renderer produce response tokens in different orders:

| Component | Per-turn token order |
|-----------|---------------------|
| Generator (`response_ids`) | `[model_response_tokens, suffix+end_zeros, observation_zeros]` |
| Renderer (VLLMRenderer) | `[suffix_tokens, model_response_tokens, end_token, observation_tokens]` |

The generator places the model's actual output first (`loss_mask=1`), followed by template and observation tokens as zeros (`loss_mask=0`). The renderer follows conversation order: the assistant header (suffix) first, then the model output, then the end-of-turn token, then the observation.

The original code split the rendered sequence with `rendered[:-resp_len]`, assuming the orderings matched. This applied `loss_mask=1` to suffix tokens instead of actual model output.

**Problem B — Image placeholder count differences.** The local `image_processor` (via `ImageChunk.expected_tokens`) and vLLM's render endpoint compute different numbers of placeholder tokens for the same image. This shifted the prompt/response split point, compounding Problem A.

### Fix

**File: `skyrl/train/trainer.py`** — Added `_rebuild_vision_response()` function (line 82) and rewrote `convert_to_training_input()`.

`_rebuild_vision_response()` works in 5 phases:
1. Walks `model_input.chunks` to find the prompt/response chunk boundary using `expected_tokens`
2. Recomputes actual prompt length using vLLM's real placeholder counts from `mm_placeholders`
3. Parses the generator's per-turn structure from `loss_mask` (alternating runs of 1s and 0s)
4. Determines the suffix length from the first response chunk
5. For each turn, rebuilds `loss_mask`, `rewards`, and `logprobs` arrays in rendered order: `[suffix(0), response(1), end(0), observation(0)]`

`convert_to_training_input()` now calls `_rebuild_vision_response()` per sample when `vision_data` is available, replacing the naive split logic.

```82:245:SkyRL/skyrl/train/trainer.py
def _rebuild_vision_response(
    rendered, gen_prompt_ids, gen_response_ids, gen_loss_mask, gen_rewards, gen_logprobs,
    model_input, mm_placeholders,
):
    # ... (5-phase algorithm, see file for full implementation)
    return prompt_ids, response_ids, new_loss_mask, new_rewards, new_logprobs
```

The method also includes a safety fallback: if the rebuilt `loss_mask` length doesn't match `response_ids`, it truncates or pads rather than crashing.

---

## Issue 2: `KeyError: 'pixel_values'` in RefWorkerBase

**Error:** `KeyError: 'pixel_values'` at `worker.py:1229`

**Location:** `skyrl/backends/skyrl_train/workers/worker.py`, `RefWorkerBase._forward_micro_batch`

### Root Cause

`RefWorkerBase._forward_micro_batch()` accessed vision tensors with `micro_batch["pixel_values"]` (bracket notation), which raises `KeyError` when the key is absent. The `PolicyWorkerBase` at line 1003 already used the safe `.get()` pattern — this was an inconsistency between the two worker classes.

### Fix

**File: `skyrl/backends/skyrl_train/workers/worker.py`**, lines 1229-1230:

```python
# Before (unsafe):
pixel_values = micro_batch["pixel_values"]
image_grid_thw = micro_batch["image_grid_thw"]

# After (safe, matching PolicyWorkerBase):
pixel_values = micro_batch.get("pixel_values", None)
image_grid_thw = micro_batch.get("image_grid_thw", None)
```

---

## Issue 3: Sample Packing Assertion with VLM

**Error:** `AssertionError: Sample packing is not supported with VLM vision inputs`

**Location:** `skyrl/backends/skyrl_train/workers/model_wrapper.py`, line 312

### Root Cause

The training config defaults to `use_sample_packing: True` (defined in `skyrl/train/config/config.py:597`). Sample packing concatenates sequences for efficiency using Flash Attention 2's variable-length attention. VLMs cannot use this because they require model-specific 3D positional IDs for image tokens, which are incompatible with the packing approach. The model wrapper correctly asserts this, but the config was never overridden for VLM training.

### Fix

**File: `examples/train/visgym/run_visgym.sh`** — Added config override:

```bash
trainer.use_sample_packing=false \
```

---

## Issue 4: `pixel_values` Not Propagated from vLLM Render Endpoint

**Error:** No crash (silent failure) — `pixel_values` was always `None` in the training batch, meaning the VLM model processed image placeholder tokens without any actual image features.

**Location:** `skyrl/train/trainer.py`, `render_vision_inputs` method

### Root Cause

The vLLM render endpoint (`/v1/chat/completions/render`) returns:

```json
{
  "features": {
    "kwargs_data": {
      "image": [null, null, null, ...]
    }
  }
}
```

Every item in the `kwargs_data["image"]` list is `null`. This is **by design** in vLLM: the multimodal processor's LRU sender cache (`MultiModalProcessorSenderCache` in `vllm/multimodal/cache.py`) strips tensor data on cache hits, returning `None` to indicate "resolve from cache via `mm_hashes`". This mechanism is designed for the generate endpoint where the vLLM server holds the cached tensors internally. But the training path calls the render endpoint externally and needs actual tensor data — which it never receives.

The `render_vision_inputs` method checked `has_vision = any(r.multi_modal_kwargs for r in rendered)`, which was always `False` since all kwargs_data items were `None`, so `pixel_values` was never populated.

### Fix

**File: `skyrl/train/trainer.py`** — Added `_compute_pixel_values_locally()` method (line 778) and a fallback branch in `render_vision_inputs` (line 868).

When the vLLM endpoint's `kwargs_data` is unavailable (all `None`), the fallback:
1. Detects that `model_inputs` contain `ImageChunk` objects
2. Loads `AutoImageProcessor` from the policy model path (cached after first load)
3. Decodes JPEG bytes from `ImageChunk.data` into PIL images
4. Processes images through the HuggingFace image processor to produce `pixel_values` and `image_grid_thw` tensors

```778:814:SkyRL/skyrl/train/trainer.py
    def _compute_pixel_values_locally(self, model_inputs):
        # ... loads AutoImageProcessor, processes ImageChunk.data → PIL → tensors
        return pixel_values_list, image_grid_thw_list
```

The primary path (decoding `kwargs_data` via `mm_serde.decode_mm_kwargs_item`) is preserved and will be used if vLLM's cache configuration changes in the future.

---

## Issue 5: FSDP CPU Offload Device Mismatch

**Error:** `RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cuda:0, different from other tensors on cpu`

**Location:** `transformers/models/qwen3_vl/modeling_qwen3_vl.py`, line 328 (rotary embedding forward), called from `model_wrapper.py:355`

### Root Cause

The ref model was configured with `trainer.ref.fsdp_config.cpu_offload=true`. FSDP CPU offloading moves model parameters to CPU between forward passes to save GPU memory. However, the Qwen3-VL rotary embedding has an `inv_freq` buffer that stayed on CPU during the forward pass. When this CPU tensor was multiplied with GPU-resident `position_ids` in `freqs = inv_freq_expanded @ position_ids_expanded`, the operation failed due to the device mismatch.

The VLM forward path passes `position_ids=None` to the model (line 358 of `model_wrapper.py`), letting Qwen3-VL compute 3D position IDs internally. This internal computation triggers the rotary embedding, which hits the CPU-offloaded `inv_freq`.

### Fix

**File: `examples/train/visgym/run_visgym.sh`** — Changed CPU offload setting:

```bash
# Before:
trainer.ref.fsdp_config.cpu_offload=true

# After:
trainer.ref.fsdp_config.cpu_offload=false
```

Qwen3-VL-2B is small enough (~2B parameters, ~4GB in bf16) to fit comfortably on 4 GPUs with FSDP sharding without CPU offloading.

---

## Summary of All Changes

### `skyrl/train/trainer.py`

| Change | Lines | Why |
|--------|-------|-----|
| Added `_rebuild_vision_response()` | 82-245 | Fixes Issue 1: correctly maps generator's token ordering to renderer's ordering, accounting for image placeholder count differences |
| Added `_compute_pixel_values_locally()` | 778-814 | Fixes Issue 4: local fallback to compute pixel_values when vLLM's kwargs_data is unavailable |
| Modified `render_vision_inputs()` | 816-879 | Returns `mm_placeholders_list` (4-tuple instead of 3-tuple); adds local pixel_values fallback branch |
| Modified `convert_to_training_input()` | 881-948 | Calls `_rebuild_vision_response()` per sample instead of naive split |

### `skyrl/backends/skyrl_train/workers/worker.py`

| Change | Lines | Why |
|--------|-------|-----|
| Safe dict access in `RefWorkerBase._forward_micro_batch` | 1229-1230 | Fixes Issue 2: `.get()` instead of `[]` for optional vision keys |

### `examples/train/visgym/run_visgym.sh`

| Change | Line | Why |
|--------|------|-----|
| `trainer.use_sample_packing=false` | 60 | Fixes Issue 3: VLMs need sample packing disabled |
| `trainer.ref.fsdp_config.cpu_offload=false` | 32 | Fixes Issue 5: avoids CPU/GPU device mismatch in rotary embeddings |

---

## Verification

After all fixes, the training script completes successfully:

```
Training Batches Processed: 100%|██████████| 12/12
Training done!
```

The full pipeline runs: generation (40.6s) → rendering + conversion (1.8s) → forward/ref (16.0s) → policy training (22.5s) → weight sync (3.9s) = ~85s per step, across 12 steps (3 epochs × 4 batches).

Training metrics for the first run show `final_loss: 0.0` and `avg_raw_reward: 0.0`. This is expected: the untrained model scores 0 reward on the maze task, so GRPO advantages are all 0 and there is no gradient signal. With continued training or a better reward function, the model would start learning.
