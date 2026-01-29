# VLM Reinforcement Learning Prototype - Change Summary

This document summarizes the changes made to SkyRL to support Vision-Language Model (VLM) reinforcement learning, using the Geometry-3K dataset as the prototype example.

## Overview

The prototype adds multi-modal (image + text) support throughout the SkyRL training pipeline:
- **Dataset layer**: Extract and propagate images from prompts
- **Inference layer**: Pass multi-modal data to vLLM and retrieve processed inputs
- **Training layer**: Include multi-modal inputs in training batches for policy/ref model forward passes

**Branch commits**: 12 commits from `e79684b` to `5b3f0ed`

**Key model**: `Qwen/Qwen3-VL-2B-Instruct`

---

## Files Changed Summary

| Category | Files | Lines Changed |
|----------|-------|---------------|
| New Example | 5 files | +472 |
| Core Training | 6 files | +390 |
| Inference Engine | 3 files | +99 |
| Dataset | 3 files | +111 |
| Tests | 1 file | +504 |
| **Total** | **23 files** | **+2,315 / -107** |

---

## New Files: Geometry-3K Example

### `examples/geometry-3k/env.py`
Custom environment for evaluating geometry problem answers:
- Extracts answers from `<answer>...</answer>` tags
- Normalizes and compares against ground truth
- Supports exact match, numeric comparison, and substring matching
- Single-turn environment (done=True after first response)

### `examples/geometry-3k/geometry_3k_dataset.py`
Dataset preprocessing script:
- Source: `hiyouga/geometry3k` HuggingFace dataset
- Converts images to multi-modal prompt format
- Creates `train.parquet` and `train-dev.parquet` outputs
- Prompt template asks for `<think>` and `<answer>` tags

### `examples/geometry-3k/run_geometry_3k-dev.sh`
Development training script:
- Uses `Qwen/Qwen3-VL-2B-Instruct` VLM
- GRPO advantage estimator
- Colocated placement (all workers on same GPUs)
- Key settings: 4 GPUs, batch_size=64, max_prompt_length=1024

---

## Core Infrastructure Changes

### 1. Dataset Layer (`skyrl_train/dataset/dataset.py`)

**New functionality:**
- `convert_bytes_to_pil_image()`: Convert byte arrays to PIL images
- `extract_images_from_messages()`: Extract images from message content
- `convert_to_pil_images()`: Batch conversion of image bytes to PIL

**PromptDataset changes:**
- Added `processor` parameter for VLM AutoProcessor support
- `__getitem__()` now returns 5-tuple: `(messages, env_class, extra, uid, multi_modal_data)`
- Multi-modal data extracted from messages or top-level row fields
- vLLM-compatible format: `{"image": PIL.Image}` or `{"image": [images]}`

### 2. TensorBatch (`skyrl_train/training_batch.py`)

**Extended to support multi-modal inputs:**

Previously only supported `torch.Tensor` values. Now supports:
- `torch.Tensor` (original)
- `TensorBatch` (nested batches)
- `list` (for lists of tensors with different shapes, e.g., pixel_values)

**Methods updated:**
- `__setitem__`, `__getitem__` - Type checking for new types
- `to()` - Device transfer for nested types
- `contiguous()` - Memory layout optimization
- `repeat()`, `repeat_interleave()` - Batch replication
- `slice()`, `chunks()` - Batch splitting
- `cat()` - Batch concatenation
- `__getstate__`, `__setstate__` - Serialization/deserialization

**New TrainingInput field:**
```python
multi_modal_inputs: Optional[TensorBatch[str, List[torch.Tensor]]]
```

### 3. Inference Engine Layer

#### `inference_engines/base.py`

**InferenceEngineInput additions:**
```python
multi_modal_data: Optional[List[Dict[str, Any]]]  # Raw images
```

**InferenceEngineOutput additions:**
```python
multi_modal_data: Optional[List[Dict[str, Any]]]    # Pass-through
multi_modal_inputs: Optional[List[Dict[str, Any]]]  # Tokenized inputs
prompt_token_ids: Optional[List[List[int]]]         # Updated with image tokens
```

#### `inference_engines/vllm/vllm_engine.py`

**Key changes in `AsyncVLLMInferenceEngine._collect_outputs()`:**
1. Build `TokensPrompt` with `multi_modal_data`
2. Use vLLM's `input_preprocessor` to tokenize image data
3. Extract `pixel_values` and `image_grid_thw` from processed inputs
4. Return updated `prompt_token_ids` that include image placeholder tokens

```python
# After generation
input_preprocessor = self.llm.input_processor.input_preprocessor
input_preprocessor.clear_mm_cache()
tokenized_out = input_preprocessor.preprocess({
    'prompt': prompt_token_ids, 
    'multi_modal_data': multi_modal_data
})
multi_modal_inputs = {
    'pixel_values': image_mm_kwargs['pixel_values'].data,
    'image_grid_thw': image_mm_kwargs['image_grid_thw'].data,
}
```

#### `inference_engines/inference_engine_client.py`

- Routes `multi_modal_data` to appropriate engines
- Collects `multi_modal_inputs` and updated `prompt_token_ids` from results

### 4. Generator Layer (`skyrl_train/generators/skyrl_gym_generator.py`)

**GeneratorInput additions:**
```python
multi_modal_data: Optional[List[Dict[str, Any]]]
```

**GeneratorOutput additions:**
```python
multi_modal_data: Optional[List[Dict[str, Any]]]    # Pass-through
multi_modal_inputs: Optional[List[Dict[str, Any]]]  # From engine
```

**generate_batched():**
- Passes `multi_modal_data` to inference engine
- Retrieves `multi_modal_inputs` and updated `prompt_token_ids`
- Uses engine-provided `prompt_token_ids` (with image tokens) instead of re-tokenizing

### 5. Trainer (`skyrl_train/trainer.py`)

**Training batch creation:**
- Extracts `multi_modal_inputs` from generator output
- Converts to `TensorBatch` format via `convert_prompts_responses_to_batch_tensors()`
- Includes in `TrainingInputBatch`

**Forward pass:**
- Includes `multi_modal_inputs` in data passed to model
- Handles `TensorBatch` type in padding logic

### 6. Model Wrapper (`skyrl_train/model_wrapper.py`)

**VLM model loading:**
- Hardcoded to `Qwen3VLForConditionalGeneration` (TODO: make configurable)
- Handles composite config (vision_config + text_config) for rope settings

**Forward pass:**
```python
def forward(self, ..., multi_modal_inputs: Dict = None):
    if multi_modal_inputs is not None:
        multi_modal_inputs_tensor = {
            'pixel_values': torch.cat(multi_modal_inputs['pixel_values'], dim=0),
            'image_grid_thw': torch.stack(multi_modal_inputs['image_grid_thw']),
        }
    # Pass to model
    output = self.model(..., **multi_modal_inputs_tensor)
```

### 7. Workers (`skyrl_train/workers/worker.py`)

Both `PolicyWorkerBase` and `RefWorkerBase` updated to:
- Extract `multi_modal_inputs` from experience/micro_batch
- Pass to `self.model()` forward call

### 8. Main Entrypoint (`skyrl_train/entrypoints/main_base.py`)

**VLM detection:**
```python
def _is_vlm_model(self, model_path: str) -> bool:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return hasattr(config, "vision_config") and hasattr(config, "text_config")
```

**Processor initialization:**
- Uses `AutoProcessor` for VLM models (provides both tokenizer and image processor)
- Falls back to `AutoTokenizer` for standard LLMs

**Environment registration:**
- Registers `geometry-3k` environment in entrypoint

### 9. Utility Updates (`skyrl_train/utils/trainer_utils.py`)

**New functions:**
- `sanitize_env_extras()`: Converts PIL Images to base64 for JSON serialization
- `_sanitize_value()`: Recursive sanitization helper

**Training batch conversion:**
- `convert_prompts_responses_to_batch_tensors()` now returns `multi_modal_inputs_tensor`

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
└─────────────────────────────────────────────────────────────────────────────┘

Dataset (parquet)
    │
    │  prompt = [{"role": "user", "content": [{"type": "image", "image": bytes}, 
    │                                          {"type": "text", "text": "..."}]}]
    ▼
PromptDataset.__getitem__()
    │
    │  Returns: (messages, env_class, extra, uid, multi_modal_data)
    │           multi_modal_data = {"image": PIL.Image}
    ▼
SkyRLGymGenerator.generate_batched()
    │
    │  InferenceEngineInput(prompts, multi_modal_data)
    ▼
AsyncVLLMInferenceEngine._collect_outputs()
    │
    │  1. TokensPrompt(prompt_token_ids, multi_modal_data)
    │  2. vLLM.generate() → responses
    │  3. input_preprocessor.preprocess() → pixel_values, image_grid_thw
    │  4. Updated prompt_token_ids (with image placeholder tokens)
    │
    │  Returns: (output, multi_modal_inputs, updated_prompt_token_ids)
    ▼
RayPPOTrainer._generator_output_to_training_input()
    │
    │  convert_prompts_responses_to_batch_tensors()
    │  → TrainingInputBatch with multi_modal_inputs
    ▼
PolicyWorkerBase.update_policy() / RefWorkerBase.fwd_pass()
    │
    │  self.model(sequences, attention_mask, multi_modal_inputs=...)
    ▼
HFModelWrapper.forward()
    │
    │  pixel_values = torch.cat(multi_modal_inputs['pixel_values'])
    │  image_grid_thw = torch.stack(multi_modal_inputs['image_grid_thw'])
    │  output = self.model(..., pixel_values=..., image_grid_thw=...)
    ▼
Policy gradient update
```

---

## Known Limitations & TODOs

### Current Limitations

1. **Model hardcoding**: `Qwen3VLForConditionalGeneration` is hardcoded in model_wrapper.py
2. **Single image support**: Only tested with single image per prompt
3. **Batched mode only**: Multi-modal not supported in non-batched agent_loop path
4. **Sample packing**: Unclear if sample packing works correctly with VLM inputs

### TODOs in Code

```python
# model_wrapper.py
# TODO MM update to enable qwen 2.5, 3
# TODO (nithinc): revert this to be how we previously had

# vllm_engine.py
# TODO (nithinc): where we should tokenize mm data
# TODO (nithinc): right now this only supports images and only one image

# model_wrapper.py
# TODO (nithinc) - hardcoded for the img case
# TODO (nithinc): not sure if sample packing works with mm inputs?

# trainer.py
# TODO (nithinc)
```

### Recommended Future Work

1. **Model abstraction**: Create a VLM model factory to support multiple VLM architectures
2. **Multi-image support**: Test and validate multiple images per prompt
3. **Video support**: Extend to video inputs (Qwen2-VL supports video)
4. **Non-batched path**: Add multi-modal support for agentic/multi-turn generation
5. **Memory optimization**: Profile and optimize GPU memory for VLM training
6. **Sample packing validation**: Verify correctness with multi-modal inputs

---

## Testing

New test file: `tests/cpu/test_train_batch.py` (+504 lines)
- Tests for `TensorBatch` with nested batches and lists
- Serialization/deserialization tests
- `repeat`, `repeat_interleave`, `slice`, `cat` operations

---

## Configuration Reference

Key config options for VLM training:

```yaml
trainer:
  policy:
    model:
      path: "Qwen/Qwen3-VL-2B-Instruct"  # VLM model path
  max_prompt_length: 1024                 # Adjust for image tokens

generator:
  batched: true                           # Required for VLM
  sampling_params:
    max_generate_length: 1024
  engine_init_kwargs:
    max_model_len: 16384                  # VLM needs larger context

environment:
  env_class: geometry-3k                  # Custom VLM environment
```
