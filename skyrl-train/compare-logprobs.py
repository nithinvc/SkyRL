import argparse
import requests
from PIL import Image
from io import BytesIO

import torch
from vllm import LLM, SamplingParams
from transformers import AutoModelForImageTextToText, AutoProcessor
from skyrl_train.utils.torch_utils import logprobs_from_logits
from skyrl_train.dataset.dataset import PromptDataset

from vllm import TokensPrompt

def vl_input(processor, tokenizer):
    """Generate input for vision-language models from geometry_3k dataset."""
    dataset = PromptDataset(
        datasets='/home/ray/data/geometry_3k/train.parquet',
        tokenizer=tokenizer,
        max_prompt_length=12000,
        processor=processor,
    )
    
    # Get the first item from the dataset
    messages, env_class, extra, uid, multi_modal_data = dataset[0]
    
    return messages


def parse_args():
    parser = argparse.ArgumentParser(description="Compare logprobs for vision-language models")
    # options: "Qwen/Qwen3-VL-2B-Instruct", - VL model
    #           "Qwen/Qwen2.5-VL-3B-Instruct" - VL model
    parser.add_argument("--model_str", type=str, required=True, help="Model identifier")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM (use 1 to minimize numerical differences)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Dtype for model computation")
    return parser.parse_args()

def main():
    args = parse_args()
    model_str = args.model_str
    tensor_parallel_size = args.tensor_parallel_size
    dtype = args.dtype


    # Step 0. initialize vllm engine and hf model. HF model goes on GPU tensor_parallel_size + 1 (so if tp=2, gpu=3)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[dtype]

    llm = LLM(
        model=model_str,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=0.5,
        max_model_len=12000,
        enforce_eager=True,  # skip CUDA graph capture for faster startup
    )

    hf_device = torch.device(f"cuda:{tensor_parallel_size + 1}")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        model_str, torch_dtype=torch_dtype, device_map=hf_device
    )

    # Step 0.5. get the hf model processor and run vl_input through processor.apply_chat_template to get token_input_ids
    processor = AutoProcessor.from_pretrained(model_str)
    tokenizer = llm.get_tokenizer()
    messages = vl_input(processor, processor.tokenizer)
    processor_output = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    token_input_ids = processor_output["input_ids"]  # shape [1, seq_len]
    pixel_values = processor_output["pixel_values"]
    image_grid_thw = processor_output["image_grid_thw"]

    # Step 1. Run llm.generate to get responses from vllm. Get the response log probabilities.
    # vllm gets passed multi_modal_data (so the pillow object)
    text = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    # Extract the pillow image from messages
    images = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for content in msg["content"]:
                if isinstance(content, dict) and content.get("type") == "image":
                    images.append(content["image"])

    sampling_params = SamplingParams(logprobs=1, max_tokens=512)
    vllm_input = TokensPrompt(prompt_token_ids=text, multi_modal_data={"image": images[0]})
    outputs = llm.generate(vllm_input, sampling_params=sampling_params)

    resp = outputs[0].outputs[0]
    response_token_ids = list(resp.token_ids)

    vllm_logprobs_list = []
    if resp.logprobs:
        for i, token_logprobs in enumerate(resp.logprobs):
            token_id = resp.token_ids[i]
            vllm_logprobs_list.append(token_logprobs[token_id].logprob)
    vllm_logprobs = torch.tensor(vllm_logprobs_list)

    # Step 2. Aggregate the output tokens and create a new full sequence
    resp_tensor = torch.tensor(response_token_ids, dtype=token_input_ids.dtype, device=hf_device)[None, :]
    full_sequence = torch.cat([token_input_ids.to(hf_device), resp_tensor], dim=1)
    attention_mask = torch.ones_like(full_sequence)

    # Step 3. compute the forward pass with the new full sequence, using the processor multi_modal outputs (e.g., pixel_values and image_grid_thw)
    with torch.inference_mode():
        hf_outputs = hf_model(
            input_ids=full_sequence,
            attention_mask=attention_mask,
            pixel_values=pixel_values.to(hf_device),
            image_grid_thw=image_grid_thw.to(hf_device),
        )

    # Step 4. compute the logprobs from the hf model forward pass - store in a tensor called hf_logprobs
    hf_logits = hf_outputs.logits[:, -len(response_token_ids) - 1:-1]
    seq_rolled = torch.roll(full_sequence, shifts=-1, dims=1)[:, -len(response_token_ids) - 1:-1]
    hf_logprobs = logprobs_from_logits(hf_logits, seq_rolled)


    hf_logprobs = hf_logprobs.cpu()
    vllm_logprobs = vllm_logprobs.cpu()

    diff = hf_logprobs - vllm_logprobs
    diff = diff.exp().abs()
    print('DTYPE:', dtype)
    print('DIFF:', diff.mean().item())
    print('DIFF STD:', diff.std().item())
    print('DIFF MAX:', diff.max().item())
    print('DIFF MIN:', diff.min().item())
    print('DIFF MEAN:', diff.mean().item())

if __name__ == "__main__":
    main()