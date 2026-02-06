# Vision-Language Reinforcement Learning Proposal for SkyRL

## Summary

This proposal outlines the design for production-ready Vision-Language Model (VLM) reinforcement learning in SkyRL. The current prototype (branch `nithinc/vlm-v2`) demonstrates end-to-end feasibility with `Qwen/Qwen3-VL-2B-Instruct` on the Geometry-3K dataset. However, the implementation requires architectural refinements for production use.

**Key Design Decisions:**

1. **HuggingFace Processor for Training**: Use `AutoProcessor` to create tokenized inputs (`prompt_token_ids` with image placeholders) and multi-modal tensors (`pixel_values`, `image_grid_thw`) for the training forward pass. The comparison report confirms HF and vLLM preprocessors produce identical log probabilities.

2. **Raw Images to vLLM**: Continue passing raw `multi_modal_data` (PIL images) to vLLM for inference. vLLM handles its own internal preprocessing - we don't modify this.

3. **Per-Element Processing**: Process each batch element individually rather than batched. The `pixel_values` tensor is packed (variable-length per image), making batch reassembly complex in async inference and training sharding scenarios.

4. **Model Abstraction**: Replace hardcoded `Qwen3VLForConditionalGeneration` with a configurable VLM model factory.

---

## Background: Current Implementation

The prototype introduces VLM support across the pipeline:

| Layer | Changes |
|-------|---------|
| Dataset | Extract images from messages, pass `multi_modal_data` |
| Inference | Pass images to vLLM, retrieve `pixel_values` from preprocessor |
| Training | Include `multi_modal_inputs` in forward pass |

### Current Data Flow

```
Dataset → multi_modal_data (PIL Images)
    ↓
vLLM.generate(prompt_token_ids, multi_modal_data)
    ↓
vLLM input_preprocessor → pixel_values, image_grid_thw, updated prompt_token_ids
    ↓
Training forward pass with multi_modal_inputs
```

### Key Finding: Processor Equivalence

From `comparison-results/Qwen_Qwen3-VL-2B-Instruct_report.txt`:

```
Comparison C: HF (HF processor) vs HF (vLLM processor)
  Identical (bitwise): True
  Max absolute diff: 0.000000e+00
```

**Conclusion**: The HuggingFace processor and vLLM's internal preprocessor produce identical results. We can safely use HF processor for training input preparation while letting vLLM handle its own preprocessing for inference.

---

## Proposed Architecture

### 1. Processor-Based Input Preparation

**Current Problem**: The prototype calls vLLM's internal `input_preprocessor.preprocess()` after generation to extract `pixel_values`. This couples training input preparation to vLLM internals.

**Solution**: Use HuggingFace `AutoProcessor` directly in the dataset/generator layer:

```python
# skyrl_train/dataset/dataset.py (proposed change)
class PromptDataset:
    def __init__(self, ..., processor: AutoProcessor = None):
        self.processor = processor
        
    def __getitem__(self, item):
        messages = row_dict.pop(self.prompt_key)
        images = extract_images_from_messages(messages)
        
        if self.processor is not None and images:
            # Use HF processor to create tokenized prompt WITH image placeholders
            processed = self.processor(
                text=self.processor.apply_chat_template(messages, add_generation_prompt=True),
                images=images,
                return_tensors="pt"
            )
            prompt_token_ids = processed["input_ids"][0].tolist()
            multi_modal_inputs = {
                "pixel_values": processed["pixel_values"],
                "image_grid_thw": processed["image_grid_thw"],
            }
        else:
            prompt_token_ids = self.tokenizer.apply_chat_template(messages, ...)
            multi_modal_inputs = None
            
        # Raw images for vLLM inference
        multi_modal_data = {"image": images[0]} if images else None
        
        return messages, env_class, extra, uid, multi_modal_data, prompt_token_ids, multi_modal_inputs
```

### 2. Per-Element Processing Rationale

**Why not batch processing?**

The `pixel_values` tensor from VLM processors is **packed** - images with different resolutions produce tensors of different shapes that are concatenated:

```python
# Example: Two images with different grid sizes
# Image 1: pixel_values shape (1024, 1176)  - 4x4 grid
# Image 2: pixel_values shape (2048, 1176)  - 8x4 grid
# Packed: pixel_values shape (3072, 1176)   - concatenated

# image_grid_thw tracks the boundaries:
# [[1, 4, 4], [1, 8, 4]]
```

**Problems with batched processing:**

1. **Async inference**: When responses arrive out-of-order, reassembling the packed tensor requires tracking which slices belong to which sample
2. **Training sharding**: When splitting batches across DP workers, we must correctly slice the packed tensor and update `image_grid_thw` indices
3. **Variable image counts**: Samples may have 0, 1, or N images - batching requires padding/masking logic

**Solution**: Process each sample individually, store `pixel_values` as a list of tensors in `TensorBatch`:

```python
# skyrl_train/training_batch.py (current implementation)
class TrainingInput(TypedDict, total=False):
    # ... existing fields ...
    multi_modal_inputs: Optional[TensorBatch[str, List[torch.Tensor]]]
    
# Each element is independent:
multi_modal_inputs = TensorBatch({
    "pixel_values": [tensor1, tensor2, tensor3, ...],  # List, not stacked
    "image_grid_thw": [grid1, grid2, grid3, ...],
})
```

This approach:
- Simplifies async result collection (each result is independent)
- Makes chunking/slicing trivial (just slice the list)
- Handles variable image counts naturally

### 3. Inference Flow (Unchanged for vLLM)

vLLM continues to receive raw images and handles preprocessing internally:

```python
# skyrl_train/inference_engines/vllm/vllm_engine.py
async def _collect_outputs(self, prompt_token_ids, ..., multi_modal_data):
    # Pass raw images to vLLM - it preprocesses internally
    prompt_input = TokensPrompt(
        prompt_token_ids=prompt_token_ids,
        multi_modal_data=multi_modal_data  # PIL images
    )
    
    async for request_output in self.llm.generate(prompt=prompt_input, ...):
        final_output = request_output
        
    # For training, we DON'T need to call input_preprocessor anymore
    # The HF processor already created multi_modal_inputs in the dataset layer
    return final_output
```

### 4. Model Wrapper Abstraction

**Current Problem** (lines 97-101 of `model_wrapper.py`):

```python
# TODO MM update to enable qwen 2.5, 3
# TODO (nithinc): revert this to be how we previously had
from transformers import Qwen3VLForConditionalGeneration
model_class = Qwen3VLForConditionalGeneration
```

**Proposed Solution**:

```python
# skyrl_train/model_wrapper.py
def get_vlm_model_class(config: AutoConfig):
    """Return appropriate VLM model class based on config."""
    if hasattr(config, "vision_config"):
        # Map architecture names to model classes
        arch_map = {
            "Qwen2VLForConditionalGeneration": "transformers.Qwen2VLForConditionalGeneration",
            "Qwen3VLForConditionalGeneration": "transformers.Qwen3VLForConditionalGeneration",
            "LlavaForConditionalGeneration": "transformers.LlavaForConditionalGeneration",
            # Add more as needed
        }
        arch_name = config.architectures[0] if config.architectures else None
        if arch_name in arch_map:
            module_path, class_name = arch_map[arch_name].rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
    return AutoModelForCausalLM
```

### 5. Forward Pass Updates

The model forward pass needs to handle packed `pixel_values`:

```python
# skyrl_train/model_wrapper.py - forward method
def forward(self, sequences, num_actions, attention_mask, ..., multi_modal_inputs=None):
    if multi_modal_inputs is not None:
        # Concatenate list of pixel_values into packed tensor
        pixel_values = torch.cat(multi_modal_inputs['pixel_values'], dim=0)
        # Stack image_grid_thw (these should have consistent shape per image)
        image_grid_thw = torch.stack(multi_modal_inputs['image_grid_thw'])
        mm_kwargs = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
    else:
        mm_kwargs = {}
    
    output = self.model(sequences, attention_mask=attention_mask, ..., **mm_kwargs)
```

---

## Multi-Turn Extension

The current implementation only supports single-turn (batched) VLM generation. Extending to multi-turn requires:

### 1. Track Multi-Modal State in Agent Loop

```python
# skyrl_train/generators/skyrl_gym_generator.py
@dataclass
class AgentLoopState:
    chat_history: ConversationType
    input_ids: List[int]
    loss_mask: List[int]
    rollout_logprobs: Optional[List[float]]
    response_end_idx: Optional[int]
    done: bool
    # NEW: Track multi-modal data across turns
    multi_modal_data: Optional[Dict[str, Any]] = None
    multi_modal_inputs: Optional[Dict[str, torch.Tensor]] = None
```

### 2. Handle Images in Observations

Environment observations may include new images:

```python
async def agent_loop(self, prompt, env_class, ...):
    # Extract initial images
    initial_images = extract_images_from_messages(prompt)
    agent_loop_state.multi_modal_data = {"image": initial_images}
    
    while not agent_loop_state.done:
        # Generate with current multi-modal context
        engine_input = InferenceEngineInput(
            prompt_token_ids=[agent_loop_state.input_ids],
            multi_modal_data=[agent_loop_state.multi_modal_data],
            ...
        )
        engine_output = await self.inference_engine_client.generate(engine_input)
        
        # Environment step may return new images
        env_step_output = await env.step(output)
        new_obs = env_step_output["observations"]
        
        # Extract any new images from observations
        new_images = extract_images_from_messages(new_obs)
        if new_images:
            # Append to multi-modal context
            existing = agent_loop_state.multi_modal_data.get("image", [])
            if not isinstance(existing, list):
                existing = [existing]
            agent_loop_state.multi_modal_data["image"] = existing + new_images
```

### 3. Re-Process Multi-Modal Inputs per Turn

For accurate loss computation, re-process the full context each turn:

```python
# When building step-wise training data
if is_step_wise:
    # Use HF processor on full context to get accurate multi_modal_inputs
    full_context_processed = self.processor(
        text=tokenizer.decode(agent_loop_state.input_ids),
        images=agent_loop_state.multi_modal_data.get("image", []),
        return_tensors="pt"
    )
    per_step_output.multi_modal_inputs = {
        "pixel_values": full_context_processed["pixel_values"],
        "image_grid_thw": full_context_processed["image_grid_thw"],
    }
```

---

## Tinker API Integration

Tinker provides a Renderer abstraction that handles multi-modal input construction. Integrating with Tinker enables:

1. Standardized VLM input preparation across different model families
2. Proper loss masking via `build_supervised_example()`
3. Future compatibility with Tinker training infrastructure

### Renderer Overview (from `tinker_vision_language.md`)

```python
# Tinker's Renderer interface
class Renderer:
    def build_generation_prompt(messages, ...) -> ModelInput:
        """Produces ModelInput for sampling/inference."""
        
    def build_supervised_example(messages, ...) -> (ModelInput, weights):
        """Produces token-level weights for loss computation."""
```

`ModelInput` contains interleaved `EncodedTextChunk` and `ImageChunk`:

```python
ModelInput(chunks=[
    EncodedTextChunk(token_ids=[...]),
    ImageChunk(image_bytes=..., format="png"),
    EncodedTextChunk(token_ids=[...]),
])
```

### Integration Approach

**Option A: Adapter Layer**

Create an adapter that converts Tinker's `ModelInput` to SkyRL's `InferenceEngineInput`:

```python
# skyrl_train/utils/tinker_adapter.py
def model_input_to_inference_input(model_input: ModelInput) -> Tuple[List[int], Dict]:
    """Convert Tinker ModelInput to SkyRL inference format."""
    token_ids = []
    images = []
    
    for chunk in model_input.chunks:
        if isinstance(chunk, EncodedTextChunk):
            token_ids.extend(chunk.token_ids)
        elif isinstance(chunk, ImageChunk):
            # Convert bytes to PIL
            img = Image.open(io.BytesIO(chunk.image_bytes))
            images.append(img)
            # Add placeholder tokens (renderer should have done this)
    
    multi_modal_data = {"image": images} if images else None
    return token_ids, multi_modal_data


def build_datum_from_generator_output(
    generator_output: GeneratorOutput,
    renderer: Renderer,
) -> List[Datum]:
    """Convert SkyRL generator output to Tinker Datum for training."""
    datums = []
    for i in range(len(generator_output["prompt_token_ids"])):
        # Reconstruct messages from tokens (if needed)
        # Build ModelInput + weights via renderer
        model_input, weights = renderer.build_supervised_example(messages)
        
        datum = datum_from_model_input_weights(
            model_input=model_input,
            weights=weights,
            # Add RL-specific loss inputs
            loss_fn_inputs={
                "advantages": generator_output["advantages"][i],
                "old_log_probs": generator_output["action_log_probs"][i],
                ...
            }
        )
        datums.append(datum)
    return datums
```

**Option B: Use Renderer for Preprocessing Only**

Use Tinker's renderer purely for input construction, keeping SkyRL's training loop:

```python
# In dataset or generator layer
from tinker_cookbook import renderers
from tinker_cookbook.image_processing_utils import get_image_processor

class VLMPromptDataset(PromptDataset):
    def __init__(self, ..., model_name: str):
        super().__init__(...)
        tokenizer = tokenizer_utils.get_tokenizer(model_name)
        image_processor = get_image_processor(model_name)
        self.renderer = renderers.get_renderer(model_name, tokenizer, image_processor)
    
    def __getitem__(self, item):
        messages = ...
        
        # Use renderer to build generation prompt
        model_input = self.renderer.build_generation_prompt(messages)
        
        # Extract token IDs and images
        prompt_token_ids = []
        images = []
        for chunk in model_input.chunks:
            if hasattr(chunk, 'token_ids'):
                prompt_token_ids.extend(chunk.token_ids)
            elif hasattr(chunk, 'image_bytes'):
                images.append(Image.open(io.BytesIO(chunk.image_bytes)))
        
        multi_modal_data = {"image": images} if images else None
        return messages, env_class, extra, uid, multi_modal_data, prompt_token_ids
```

### Recommended Path

Start with **Option B** (Renderer for preprocessing) as it:
- Minimizes changes to SkyRL's training loop
- Leverages Tinker's battle-tested VLM input construction
- Provides a migration path to full Tinker integration later

---

## Implementation Checklist

### Phase 1: Core Refactoring

- [ ] Refactor `PromptDataset` to use HF processor for `prompt_token_ids` and `multi_modal_inputs`
- [ ] Remove `input_preprocessor.preprocess()` call from `vllm_engine.py` (move to dataset layer)
- [ ] Add VLM model factory in `model_wrapper.py` to replace hardcoded `Qwen3VLForConditionalGeneration`
- [ ] Update `main_base.py` VLM detection to use model factory

### Phase 2: Training Pipeline

- [ ] Validate `TensorBatch` chunking/slicing with list-based `multi_modal_inputs`
- [ ] Test training with variable-length pixel_values across DP workers
- [ ] Add sample packing validation for VLM (currently flagged as TODO)

### Phase 3: Multi-Turn Support

- [ ] Extend `AgentLoopState` with `multi_modal_data` tracking
- [ ] Update `agent_loop` to handle images in environment observations
- [ ] Test multi-turn VLM generation with image-returning environments

### Phase 4: Tinker Integration

- [ ] Implement Tinker adapter layer
- [ ] Test Renderer-based input construction
- [ ] Document migration path for existing VLM environments

---

## Appendix: File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `dataset/dataset.py` | Modify | Use processor for prompt_token_ids + multi_modal_inputs |
| `inference_engines/vllm/vllm_engine.py` | Simplify | Remove input_preprocessor calls, pass through multi_modal_data |
| `model_wrapper.py` | Modify | Add VLM model factory, remove hardcoded model class |
| `generators/skyrl_gym_generator.py` | Extend | Add multi_modal_data to AgentLoopState |
| `training_batch.py` | No change | Current list-based TensorBatch works correctly |
| `workers/worker.py` | No change | Already handles multi_modal_inputs |
| `utils/tinker_adapter.py` | New | Adapter for Tinker Renderer integration |

---

## References

- Current prototype: Branch `nithinc/vlm-v2`
- Processor comparison: `comparison-results/Qwen_Qwen3-VL-2B-Instruct_report.txt`
- Tinker VLM docs: `tinker_vision_language.md`
- vlm_changes.md: Detailed prototype change log
