# -*- coding: utf-8 -*-

from typing import List

import numpy as np
from diffusers import CogView4Pipeline


class ImageGenerationService(object):
    def __init__(self, cogview4_path: str | None) -> None:
        self._models = {}
        if cogview4_path is not None:
            self._models["cogview-4"] = CogView4Pipeline.from_pretrained(cogview4_path)

    def generate(self, model: str, prompt: str, size: int, num_images: int) -> List[np.ndarray]:
        if model not in self._models:
            raise ValueError(f"Model {model} not found")
        width, height = list(map(int,size.split('x')))
        images_list = self._models["cogview-4"](
            prompt=prompt,
            guidance_scale=3.5,
            num_images_per_prompt=num_images,
            num_inference_steps=50,
            width=width,
            height=height,
            output_type="np"
            ).images
        return images_list
