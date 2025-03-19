# -*- coding: utf-8 -*-


import numpy as np
from diffusers import CogView4Pipeline


class ImageGenerationService(object):
    def __init__(self, cogview4_path: str | None) -> None:
        self._models = {}
        if cogview4_path is not None:
            cogview4_pl = CogView4Pipeline.from_pretrained(cogview4_path)
            cogview4_pl.enable_model_cpu_offload()
            cogview4_pl.vae.enable_slicing()
            cogview4_pl.vae.enable_titling()
            self._models["cogview-4"] = cogview4_pl

    def generate(self, model: str, prompt: str, size: int, num_images: int) -> list[np.ndarray]:
        if model not in self._models:
            raise ValueError(f"Model {model} not found")
        width, height = list(map(int, size.split("x")))
        images_lst = self._models[model](
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=50,
            guidance_scale=3.5,
            num_images_per_prompt=num_images,
            output_type="np",
        ).images
        return images_lst
