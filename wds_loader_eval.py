import io
import json
from random import random

import PIL
import torch as th
import webdataset as wds

from glide_finetune.glide_util import (get_tokens_and_mask,
                                       get_uncond_tokens_mask)
from glide_finetune.train_util import pil_image_to_norm_tensor


def glide_wds_loader(
    urls,
    enable_text=True,
    enable_image=True,
    enable_metadata=True,
    image_key="jpg",
    caption_key="txt",
    metadata_key="json",
    cache_path=None,
    # tokenizer=None,
    base_x=64,
    base_y=64,
    uncond_p=0.2,
    nsfw_filter=True,
    ar_lower=0.5,
    ar_upper=2.0,
    min_original_height=256,
    min_original_width=256,
    enable_upsample=False,
    similarity_threshold_upper=0.0,
    similarity_threshold_lower=0.5,
    words_to_skip=[],
    dataset_name="cc12m",  # can be laion, alamy.
    upscale_factor=4,
):

    base_image_shape = (base_x, base_y)
    upsample_image_shape = (int(base_x * upscale_factor), int(base_y * upscale_factor))
    dataset = wds.WebDataset(
        urls,
        cache_dir=cache_path,
        cache_size=10**10,
        handler=wds.handlers.reraise_exception,
    )

    def filter_dataset_laion(item):
        if enable_text and caption_key not in item:
            return False
        if enable_image and image_key not in item:
            return False
        if enable_metadata and metadata_key not in item:
            return False

        metadata = json.loads(item["json"].decode("utf-8"))

        similarity = float(metadata["similarity"])
        original_height = float(metadata["original_height"])
        original_width = float(metadata["original_width"])
        aspect_ratio = original_width / original_height
        caption = item[caption_key].decode("utf-8").lower()
        nsfw_rating = metadata["NSFW"]

        if original_height < min_original_height or original_width < min_original_width:
            return False
        if aspect_ratio < ar_lower or aspect_ratio > ar_upper:
            return False
        if (
            similarity < similarity_threshold_lower
            or similarity > similarity_threshold_upper
        ):
            return False
        if nsfw_filter and nsfw_rating in ["NSFW", "LIKELY"]:
            return False
        if any(slur.lower() in caption for slur in words_to_skip):
            return False
        return True

    def filter_dataset_alamy(item):
        if enable_image and "jpg" not in item:
            return False
        if enable_metadata and "json" not in item:
            return False
        metadata = json.loads(item["json"].decode("utf-8"))
        language_code = metadata["lc"]
        if language_code != "en":
            return False
        if enable_text and "caption" not in metadata:
            return False
        return True  # all good

    def filter_dataset_cc12m(item):
        if enable_image and "jpg" not in item:
            return False
        if enable_text and "txt" not in item:
            return False
        return True

    dataset_name = "cc12m"
    print("DATASET NAME: ", dataset_name)

    if dataset_name == "laion":
        filtered_dataset = dataset.select(filter_dataset_laion)
    elif dataset_name == "alamy":
        filtered_dataset = dataset.select(filter_dataset_alamy)
    elif dataset_name == "cc12m":
        filtered_dataset = dataset.select(filter_dataset_cc12m)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Must be one of 'laion' or 'alamy' or 'cc12m'."
        )

    def preprocess_dataset_FID(item):
        ''' Store all captions in a .csv file. We'll use these to generate
            synthetic images. Store real, resized (64x64) images in a
            directory.
        '''

        caption = item[caption_key].decode("utf-8")
        image_data = item[image_key]
        original_pil_image = PIL.Image.open(io.BytesIO(image_data))

        # Resize to 64x64
        base_pil_image = original_pil_image.resize(base_image_shape, resample=PIL.Image.BICUBIC).convert("RGB")
        #base_tensor = pil_image_to_norm_tensor(base_pil_image)


        return caption, base_pil_image

    transformed_dataset = filtered_dataset.map(
        preprocess_dataset_FID, handler=wds.handlers.reraise_exception
    )
    return transformed_dataset
