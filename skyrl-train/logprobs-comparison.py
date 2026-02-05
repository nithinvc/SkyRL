import argparse
import requests
from PIL import Image
from io import BytesIO


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


    # Step 0. initialize vllm engine and hf model. HF model goes on GPU 3 tensor_parallel_size + 1 (so if tp =2, gpu=3)

    # step 0.5 get the hf model processor and run vl_input through processor.apply_chat_template to get token_input_ids 

    # Step 1. Run llm.generate to get responses from vllm. Get the response log probabilities. vllm gets passed multi_modal_data (so the pillow object)

    # Step 2. Aggregate the output tokens and create a new full sequence

    # step 3. compute the forward pass with the new full sequence, using the processor multi_modal outputs (e.g., pixel_values and image_grid_thw)

    # step 4. compute the logprobs from the hf model forward pass