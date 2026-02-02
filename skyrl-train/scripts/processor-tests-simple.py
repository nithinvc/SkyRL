from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
import argparse
from typing import List, Tuple, Dict, Any, Optional
import os

from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import requests
from io import BytesIO

import torch

from skyrl_train.utils.torch_utils import logprobs_from_logits


def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, name: str) -> Dict[str, Any]:
    """Compare two tensors and return detailed comparison metrics."""
    result = {
        "name": name,
        "t1_shape": tuple(t1.shape),
        "t2_shape": tuple(t2.shape),
        "shapes_match": t1.shape == t2.shape,
    }
    
    if t1.shape != t2.shape:
        result["identical"] = False
        result["error"] = "Shape mismatch"
        return result
    
    # Convert to same dtype for comparison
    t1_f = t1.float()
    t2_f = t2.float()
    
    result["identical"] = torch.equal(t1, t2)
    result["allclose_default"] = torch.allclose(t1_f, t2_f)
    result["allclose_rtol1e3"] = torch.allclose(t1_f, t2_f, rtol=1e-3, atol=1e-5)
    result["allclose_rtol1e2"] = torch.allclose(t1_f, t2_f, rtol=1e-2, atol=1e-4)
    
    diff = (t1_f - t2_f).abs()
    result["max_abs_diff"] = diff.max().item()
    result["mean_abs_diff"] = diff.mean().item()
    result["median_abs_diff"] = diff.median().item()
    
    # Relative difference (avoid division by zero)
    denom = torch.clamp(t2_f.abs(), min=1e-10)
    rel_diff = diff / denom
    result["max_rel_diff"] = rel_diff.max().item()
    result["mean_rel_diff"] = rel_diff.mean().item()
    
    return result


def format_comparison_result(result: Dict[str, Any]) -> str:
    """Format a comparison result dictionary as a readable string."""
    lines = [f"  {result['name']}:"]
    lines.append(f"    Shape t1: {result['t1_shape']}, Shape t2: {result['t2_shape']}")
    lines.append(f"    Shapes match: {result['shapes_match']}")
    
    if not result['shapes_match']:
        lines.append(f"    ERROR: {result.get('error', 'Unknown')}")
        return "\n".join(lines)
    
    lines.append(f"    Identical (bitwise): {result['identical']}")
    lines.append(f"    Close (default rtol=1e-5, atol=1e-8): {result['allclose_default']}")
    lines.append(f"    Close (rtol=1e-3, atol=1e-5): {result['allclose_rtol1e3']}")
    lines.append(f"    Close (rtol=1e-2, atol=1e-4): {result['allclose_rtol1e2']}")
    lines.append(f"    Max absolute diff: {result['max_abs_diff']:.6e}")
    lines.append(f"    Mean absolute diff: {result['mean_abs_diff']:.6e}")
    lines.append(f"    Median absolute diff: {result['median_abs_diff']:.6e}")
    lines.append(f"    Max relative diff: {result['max_rel_diff']:.6e}")
    lines.append(f"    Mean relative diff: {result['mean_rel_diff']:.6e}")
    return "\n".join(lines)

def vl_input():
    """Generate input for vision-language models."""
    url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    img = Image.open(BytesIO(resp.content))
    img.load()  # fully load/decode the image

    print(f"Image size: {img.size}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    return messages


def parse_args():
    parser = argparse.ArgumentParser(description="Compare HF processor outputs with vLLM preprocessor")
    # options: "Qwen/Qwen3-VL-2B-Instruct", - VL model
    #           "allenai/Molmo2-4B" - VL model
    parser.add_argument("--model_str", type=str, required=True, help="Model identifier")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallel size for vLLM")
    parser.add_argument("--output_dir", type=str, default="./comparison-results", help="Directory to save results")
    return parser.parse_args()


def extract_images_from_messages(messages: List[Dict]) -> List[Any]:
    """Extract image data (URLs or objects) from message content."""
    images = []
    for msg in messages:
        if "content" in msg and isinstance(msg["content"], list):
            for content in msg["content"]:
                if isinstance(content, dict) and content.get("type") == "image":
                    image_data = content.get("image")
                    if image_data is not None:
                        images.append(image_data)
    return images


def vllm_rollout_and_logprobs(llm: LLM, messages: List[Dict]) -> Tuple[List[int], List[float]]:
    """
    Run vLLM generation and collect output response tokens and their log probabilities.
    
    Args:
        llm: The vLLM LLM instance
        messages: Chat messages in OpenAI format
        
    Returns:
        Tuple of (response_token_ids, logprobs) where:
        - response_token_ids: List of token IDs for the generated response
        - logprobs: List of log probabilities for each token
    """
    # Enable logprobs in sampling params (logprobs=1 returns the top token's logprob)
    sampling_params = SamplingParams(logprobs=1, max_tokens=512)
    
    # Get the tokenizer and apply chat template
    tokenizer = llm.get_tokenizer()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Extract multi-modal data from messages
    images = extract_images_from_messages(messages)
    
    # Build multi_modal_data dict
    multi_modal_data = {}
    if images:
        multi_modal_data["image"] = images[0] if len(images) == 1 else images

    # Use generate with text prompt and multi_modal_data
    vllm_inputs = [{
        "prompt": text,
        "multi_modal_data": multi_modal_data if multi_modal_data else None,
    }]
    
    outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
    
    # Extract from the first output (we only have one prompt)
    output = outputs[0]
    resp = output.outputs[0]
    
    # Get response token IDs
    response_tokens = list(resp.token_ids)
    
    # Extract logprobs for each token
    logprobs = []
    if resp.logprobs:
        for i, token_logprobs in enumerate(resp.logprobs):
            # token_logprobs is a dict mapping token_id -> Logprob object
            token_id = resp.token_ids[i]
            logprob = token_logprobs[token_id].logprob
            logprobs.append(logprob)
    
    return response_tokens, logprobs


def vllm_preprocess(llm: LLM, messages: List[Dict]) -> Dict[str, Any]:
    """
    Run vLLM's input preprocessor to get multi-modal kwargs.
    
    Clears the mm cache and returns the mm_kwargs dict from the preprocessor output.
    
    Args:
        llm: The vLLM LLM instance
        messages: Chat messages in OpenAI format
        
    Returns:
        Dict containing mm_kwargs (e.g., pixel_values, image_grid_thw for images)
    """
    # Get the input preprocessor from the llm engine
    input_preprocessor = llm.llm_engine.input_processor.input_preprocessor
    
    # Clear the mm cache to avoid caching issues
    input_preprocessor.clear_mm_cache()
    
    # Get the tokenizer/processor from the llm
    tokenizer = llm.get_tokenizer()
    
    # Apply chat template to get the text prompt
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize to get prompt_token_ids
    prompt_token_ids = tokenizer.encode(text)
    
    # Extract multi-modal data from messages
    images = extract_images_from_messages(messages)
    
    # Build multi_modal_data dict
    multi_modal_data = {}
    if images:
        multi_modal_data["image"] = images[0] if len(images) == 1 else images
    
    # Run the preprocessor
    tokenized_out = input_preprocessor.preprocess({
        'prompt': prompt_token_ids,
        'multi_modal_data': multi_modal_data
    })
    
    # Extract mm_kwargs from the output
    mm_kwargs = {}
    
    # Helper function to extract data from vLLM's MultiModalKwargsItem
    def extract_tensor(mm_kwargs_item, key):
        """Extract tensor from MultiModalKwargsItem using dict-style access."""
        if key not in mm_kwargs_item:
            return None
        item = mm_kwargs_item[key]
        # Handle NestedTensors wrapper (has .data attribute)
        if hasattr(item, 'data'):
            return item.data
        return item
    
    # Handle image inputs
    if 'mm_kwargs' in tokenized_out and 'image' in tokenized_out['mm_kwargs']:
        image_mm_kwargs = tokenized_out['mm_kwargs']['image'][0]
        pixel_values = extract_tensor(image_mm_kwargs, 'pixel_values')
        if pixel_values is not None:
            mm_kwargs['pixel_values'] = pixel_values
        image_grid_thw = extract_tensor(image_mm_kwargs, 'image_grid_thw')
        if image_grid_thw is not None:
            mm_kwargs['image_grid_thw'] = image_grid_thw
    
    return mm_kwargs


@torch.inference_mode()
def main():
    args = parse_args()
    llm = LLM(model=args.model_str, tensor_parallel_size=args.tensor_parallel_size)
    messages = vl_input()

    # list of token ids, logprobs for each of those tokens
    print("=" * 60)
    print("Running vLLM rollout and collecting logprobs...")
    response_tokens, logprobs = vllm_rollout_and_logprobs(llm, messages)
    logprobs = torch.tensor(logprobs)
    
    print(f"\nResponse token count: {len(response_tokens)}")
    print(f"Logprobs count: {len(logprobs)}")
    print(f"First 10 response tokens: {response_tokens[:10]}")
    print(f"First 10 logprobs: {logprobs[:10]}")
    
    # Decode the response for display
    tokenizer = llm.get_tokenizer()
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    print(f"\nGenerated response:\n{response_text}")
    
    print("\n" + "=" * 60)
    print("Running vLLM preprocessing to get mm_kwargs...")
    processed_inputs = vllm_preprocess(llm, messages)

    print(f"\nProcessed inputs keys: {list(processed_inputs.keys())}")
    for key, value in processed_inputs.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}")
    
    print("\n" + "=" * 60)
    print("Running HF with HF processor inputs...")
    model_dev = torch.device("cuda:3")
    model = AutoModelForImageTextToText.from_pretrained(args.model_str).to(model_dev)
    processor = AutoProcessor.from_pretrained(args.model_str)
    model_inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    model_inputs = {k: v.to(model_dev) if isinstance(v, torch.Tensor) else v for k, v in model_inputs.items()}
    # add resp tokens\
    input_ids = model_inputs['input_ids']
    resp_tokens_tensor = torch.tensor(response_tokens, dtype=input_ids.dtype, device=input_ids.device)[None, :]
    model_inputs['input_ids'] = torch.cat([input_ids, resp_tokens_tensor], dim=1)
    # update attn mask - diff from causal mask
    if 'attention_mask' in model_inputs and (model_inputs['attention_mask'] == 1).all():
        old_dtype = model_inputs['attention_mask'].dtype
        model_inputs['attention_mask'] = torch.ones_like(model_inputs['input_ids'], dtype=old_dtype, device=input_ids.device)
    else:
        raise ValueError("Attention mask is not all ones")

    hf_processor_outputs = model(**model_inputs)
    hf_logits = hf_processor_outputs.logits[:, -len(response_tokens) - 1:-1]
    seq_rolled = torch.roll(model_inputs['input_ids'], shifts=-1, dims=1)
    seq_rolled = seq_rolled[:, -len(response_tokens) - 1:-1]
    hf_log_probs = logprobs_from_logits(hf_logits, seq_rolled)


    # now with vllm inputs
    from copy import deepcopy
    hf_vllm_inputs = deepcopy(model_inputs)
    vllm_pixel_values = torch.tensor(processed_inputs['pixel_values'], device=model_dev)
    vllm_image_grid_thw = torch.tensor(processed_inputs['image_grid_thw'], device=model_dev)
    hf_vllm_inputs['pixel_values'] = vllm_pixel_values[None, :]
    hf_vllm_inputs['image_grid_thw'] = vllm_image_grid_thw[None, :]
    hf_vllm_outputs = model(**hf_vllm_inputs)
    hf_vllm_logits = hf_vllm_outputs.logits[:, -len(response_tokens) - 1:-1]
    hf_vllm_log_probs = logprobs_from_logits(hf_vllm_logits, seq_rolled, inplace_backward=False)

    # ============================================================
    # COMPARISON SECTION
    # ============================================================
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(f"PROCESSOR COMPARISON REPORT: {args.model_str}")
    report_lines.append("=" * 70)
    report_lines.append("")

    # ----------------------------------------------------------
    # 1. Compare pixel_values and image_grid_thw
    # ----------------------------------------------------------
    report_lines.append("-" * 70)
    report_lines.append("1. PIXEL VALUES AND IMAGE GRID THW COMPARISON")
    report_lines.append("   (vLLM preprocessor vs HuggingFace processor)")
    report_lines.append("-" * 70)
    
    # Get HF processor pixel_values and image_grid_thw
    hf_pixel_values = model_inputs['pixel_values'].squeeze(0)  # Remove batch dim for comparison
    hf_image_grid_thw = model_inputs['image_grid_thw'].squeeze(0)  # Remove batch dim
    
    # Compare pixel_values
    pv_comparison = compare_tensors(vllm_pixel_values.cpu(), hf_pixel_values.cpu(), "pixel_values")
    report_lines.append(format_comparison_result(pv_comparison))
    report_lines.append("")
    
    # Compare image_grid_thw
    thw_comparison = compare_tensors(vllm_image_grid_thw.cpu(), hf_image_grid_thw.cpu(), "image_grid_thw")
    report_lines.append(format_comparison_result(thw_comparison))
    report_lines.append("")

    # Summary for pixel_values/image_grid_thw
    report_lines.append("  SUMMARY:")
    if pv_comparison['identical'] and thw_comparison['identical']:
        report_lines.append("    pixel_values and image_grid_thw are IDENTICAL between vLLM and HF")
    elif pv_comparison.get('allclose_default', False) and thw_comparison.get('allclose_default', False):
        report_lines.append("    pixel_values and image_grid_thw are CLOSE (within default tolerance)")
    else:
        report_lines.append("    pixel_values and/or image_grid_thw DIFFER between vLLM and HF")
        if not pv_comparison.get('shapes_match', False):
            report_lines.append(f"      - pixel_values shapes differ: vLLM={pv_comparison['t1_shape']}, HF={pv_comparison['t2_shape']}")
        if not thw_comparison.get('shapes_match', False):
            report_lines.append(f"      - image_grid_thw shapes differ: vLLM={thw_comparison['t1_shape']}, HF={thw_comparison['t2_shape']}")
    report_lines.append("")

    # ----------------------------------------------------------
    # 2. Compare log probabilities between the 3 methods
    # ----------------------------------------------------------
    report_lines.append("-" * 70)
    report_lines.append("2. LOG PROBABILITIES COMPARISON")
    report_lines.append("   Comparing: vLLM vs HF+HF_processor vs HF+vLLM_processor")
    report_lines.append("-" * 70)
    
    # Ensure all logprobs are on CPU for comparison
    vllm_logprobs_cpu = logprobs.cpu().float()
    hf_logprobs_cpu = hf_log_probs.squeeze(0).cpu().float()  # Remove batch dim
    hf_vllm_logprobs_cpu = hf_vllm_log_probs.squeeze(0).cpu().float()  # Remove batch dim
    
    report_lines.append(f"  Shapes:")
    report_lines.append(f"    vLLM logprobs: {vllm_logprobs_cpu.shape}")
    report_lines.append(f"    HF (HF processor) logprobs: {hf_logprobs_cpu.shape}")
    report_lines.append(f"    HF (vLLM processor) logprobs: {hf_vllm_logprobs_cpu.shape}")
    report_lines.append("")

    # Compare 1: vLLM vs HF (HF processor)
    report_lines.append("  Comparison A: vLLM vs HF (with HF processor)")
    lp_vllm_vs_hf = compare_tensors(vllm_logprobs_cpu, hf_logprobs_cpu, "vLLM vs HF+HF_proc")
    report_lines.append(format_comparison_result(lp_vllm_vs_hf))
    report_lines.append("")

    # Compare 2: vLLM vs HF (vLLM processor)
    report_lines.append("  Comparison B: vLLM vs HF (with vLLM processor)")
    lp_vllm_vs_hf_vllm = compare_tensors(vllm_logprobs_cpu, hf_vllm_logprobs_cpu, "vLLM vs HF+vLLM_proc")
    report_lines.append(format_comparison_result(lp_vllm_vs_hf_vllm))
    report_lines.append("")

    # Compare 3: HF (HF processor) vs HF (vLLM processor)
    report_lines.append("  Comparison C: HF (HF processor) vs HF (vLLM processor)")
    lp_hf_vs_hf_vllm = compare_tensors(hf_logprobs_cpu, hf_vllm_logprobs_cpu, "HF+HF_proc vs HF+vLLM_proc")
    report_lines.append(format_comparison_result(lp_hf_vs_hf_vllm))
    report_lines.append("")

    # Summary for logprobs
    report_lines.append("  LOGPROBS SUMMARY:")
    
    def summarize_match(result: Dict, name: str) -> str:
        if result.get('identical'):
            return f"    {name}: IDENTICAL"
        elif result.get('allclose_default'):
            return f"    {name}: CLOSE (default tolerance)"
        elif result.get('allclose_rtol1e3'):
            return f"    {name}: CLOSE (rtol=1e-3)"
        elif result.get('allclose_rtol1e2'):
            return f"    {name}: CLOSE (rtol=1e-2)"
        else:
            return f"    {name}: DIFFERENT (max_abs_diff={result.get('max_abs_diff', 'N/A'):.6e})"
    
    report_lines.append(summarize_match(lp_vllm_vs_hf, "vLLM vs HF+HF_proc"))
    report_lines.append(summarize_match(lp_vllm_vs_hf_vllm, "vLLM vs HF+vLLM_proc"))
    report_lines.append(summarize_match(lp_hf_vs_hf_vllm, "HF+HF_proc vs HF+vLLM_proc"))
    report_lines.append("")
    
    # Determine which pairs are the same/different
    report_lines.append("  CONCLUSION:")
    
    # Check closeness with a reasonable tolerance for neural network outputs
    tolerance_check = lambda r: r.get('allclose_rtol1e3', False) or r.get('allclose_default', False) or r.get('identical', False)
    
    vllm_hf_same = tolerance_check(lp_vllm_vs_hf)
    vllm_hf_vllm_same = tolerance_check(lp_vllm_vs_hf_vllm)
    hf_hf_vllm_same = tolerance_check(lp_hf_vs_hf_vllm)
    
    if vllm_hf_same and vllm_hf_vllm_same and hf_hf_vllm_same:
        report_lines.append("    All three methods produce EQUIVALENT logprobs.")
    else:
        if vllm_hf_same:
            report_lines.append("    - vLLM and HF+HF_processor produce EQUIVALENT logprobs")
        else:
            report_lines.append("    - vLLM and HF+HF_processor produce DIFFERENT logprobs")
            
        if vllm_hf_vllm_same:
            report_lines.append("    - vLLM and HF+vLLM_processor produce EQUIVALENT logprobs")
        else:
            report_lines.append("    - vLLM and HF+vLLM_processor produce DIFFERENT logprobs")
            
        if hf_hf_vllm_same:
            report_lines.append("    - HF+HF_processor and HF+vLLM_processor produce EQUIVALENT logprobs")
        else:
            report_lines.append("    - HF+HF_processor and HF+vLLM_processor produce DIFFERENT logprobs")
    
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)

    # Print report to console
    full_report = "\n".join(report_lines)
    print("\n" + full_report)

    # Save report to file
    os.makedirs(args.output_dir, exist_ok=True)
    # Sanitize model name for filename (replace / with _)
    model_str_safe = args.model_str.replace("/", "_")
    report_path = os.path.join(args.output_dir, f"{model_str_safe}_report.txt")
    with open(report_path, "w") as f:
        f.write(full_report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
