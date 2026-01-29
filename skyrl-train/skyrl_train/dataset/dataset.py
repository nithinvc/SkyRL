import datasets
from loguru import logger
import os
from typing import List
from transformers import PreTrainedTokenizerBase, AutoProcessor
from PIL import Image
import io
from typing import Any, Dict
from copy import deepcopy

def convert_bytes_to_pil_image(bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(bytes)) as im:
        new_im = im.copy()
    return new_im


def extract_images_from_messages(messages: List[Dict[str, Any]]) -> List[Any]:
    """Extract images from message content.
    Images can be stored in two ways:
    1. In message content as {"type": "image", "image": <PIL.Image>}
    2. As a top-level "image" or "images" field in the row
    Args:
        messages: List of message dicts with 'role' and 'content' keys
    Returns:
        List of PIL Images found in the messages
    """
    images = []
    for message in messages:
        content = message.get("content", [])
        # Content can be a string (text-only) or a list (multi-modal)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    # Image is stored in the "image" key
                    img = item.get("image")
                    if img is not None:
                        images.append(deepcopy(img))
                        # convert this image to pillow
                        item["image"] = convert_bytes_to_pil_image(img["bytes"])

    return images


def convert_to_pil_images(images: List[Any]) -> List[Image.Image]:
    pil_images = []
    for image in images:
        b = image["bytes"]
        assert b is not None, "found none bytes"
        with Image.open(io.BytesIO(b)) as im:
            new_im = im.copy()  # copy so the underlying buffer can be released
        pil_images.append(new_im)
    return pil_images


class PromptDataset:
    def __init__(
        self,
        datasets: str | List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        num_workers: int = 8,
        prompt_key: str = "prompt",
        env_class_key: str = "env_class",
        processor: AutoProcessor = None,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompt_key = prompt_key
        self.env_class_key = env_class_key
        self.num_workers = num_workers

        self.datasets = datasets
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]

        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        loaded_datasets = []
        for source in self.datasets:
            ext = os.path.splitext(source)[-1].lower()
            if ext == ".parquet":
                ds = datasets.load_dataset("parquet", data_files=source, keep_in_memory=True)["train"]
            elif ext in [".json", ".jsonl"]:
                ds = datasets.load_dataset("json", data_files=source, keep_in_memory=True)["train"]
            else:
                # Treat as HF dataset spec: "name" or "name:split"
                dataset_name, has_split, split = source.partition(":")
                try:
                    ds_dict = datasets.load_dataset(path=dataset_name, keep_in_memory=True)
                except ValueError:
                    raise ValueError(f"Dataset `{dataset_name}` not found on Hugging Face.")
                split = split if has_split else "train"
                if split not in ds_dict:
                    raise ValueError(
                        f"Split `{split}` not found in dataset `{dataset_name}`. Configured split was `{split}` and default is `train`"
                    )
                ds = ds_dict[split]
            loaded_datasets.append(ds)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(loaded_datasets)

        logger.info(f"Total dataset size: {len(self.dataframe)}")

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        self.dataframe = self.dataframe.filter(
            lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))
            <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

        logger.info(f"Filtered dataset size: {len(self.dataframe)}")

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        messages = row_dict.pop(self.prompt_key)
        env_class = row_dict.pop(self.env_class_key, None)

        extra = {key: value for key, value in row_dict.items() if key != self.prompt_key and key != self.env_class_key}
        uid = str(item)
        multi_modal_data = {}
        if self.processor is not None:
            # Extract images from multiple possible sources:
            # 1. From message content (e.g., geometry_3k format)
            # 2. From top-level row fields (e.g., "image" or "images" column)
            images = extract_images_from_messages(messages)
            images = convert_to_pil_images(images)

            # Fallback to top-level image fields if no images found in messages
            if not images:
                if "image" in row_dict and row_dict["image"] is not None:
                    images = [row_dict["image"]]
                elif "images" in row_dict and row_dict["images"] is not None:
                    images = row_dict["images"] if isinstance(row_dict["images"], list) else [row_dict["images"]]

            # vLLM expects multi_modal_data with "image" key for single image
            # or list of images for multiple images
            if images:
                multi_modal_data = {"image": images[0]} if len(images) == 1 else {"image": images}

        return messages, env_class, extra, uid, multi_modal_data

    def collate_fn(self, item_list):
        all_inputs = []
        for prompt, env_class, env_extras, item_uids, multi_modal_data in item_list:
            all_inputs.append(
                {
                    "prompt": prompt,
                    "env_class": env_class,
                    "env_extras": env_extras,
                    "uid": item_uids,
                    "multi_modal_data": multi_modal_data,
                }
            )
        return all_inputs

    def __len__(self):
        return len(self.dataframe)
