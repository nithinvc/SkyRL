from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
import argparse
from typing import List, Tuple, Dict, Any, Optional

from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import requests
from io import BytesIO

import torch

from skyrl_train.utils.torch_utils import logprobs_from_logits

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
    hf_vllm_inputs['pixel_values'] = torch.tensor(processed_inputs['pixel_values'], device=model_dev)[None, :]
    hf_vllm_inputs['image_grid_thw'] = torch.tensor(processed_inputs['image_grid_thw'],  device=model_dev)[None, :]
    hf_vllm_outputs = model(**hf_vllm_inputs)
    hf_vllm_logits = hf_vllm_outputs.logits[:, -len(response_tokens) - 1:-1]
    hf_vllm_log_probs = logprobs_from_logits(hf_vllm_logits, seq_rolled, inplace_backward=False)

    # TODO COMPARE



if __name__ == "__main__":
    main()
