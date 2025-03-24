# -*- coding: utf-8 -*-


import numpy as np
import os

import torch
from diffusers import CogView4Pipeline

from cogkit.api.logging import get_logger
from cogkit.api.settings import APISettings

_logger = get_logger(__name__)


class ImageGenerationService(object):
    def __init__(self, settings: APISettings) -> None:
        self._models = {}
        if settings.cogview4_path is not None:
            cogview4_pl = CogView4Pipeline.from_pretrained(
                settings.cogview4_path,
                torch_dtype=torch.bfloat16 if settings.dtype == "bfloat16" else torch.float32,
            )
            if settings.offload_type == "cpu_model_offolad":
                cogview4_pl.enable_model_cpu_offload()
            else:
                cogview4_pl.to("cuda")
            cogview4_pl.vae.enable_slicing()
            cogview4_pl.vae.enable_tiling()
            self._models["cogview-4"] = cogview4_pl

        ### Check if loaded models are supported
        for model in self._models.keys():
            if model not in settings._supported_models:
                raise ValueError(
                    f"Registered model {model} not in supported list: {settings._supported_models}"
                )

        ### Check if all supported models are loaded
        for model in settings._supported_models:
            if model not in self._models:
                _logger.warning(f"Model {model} not loaded")

    @property
    def supported_models(self) -> list[str]:
        return list(self._models.keys())

    def generate(
        self,
        model: str,
        prompt: str,
        size: str,
        num_images: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        lora_path: str | None = None,
    ) -> list[np.ndarray]:
        if model not in self._models:
            raise ValueError(f"Model {model} not loaded")
        width, height = list(map(int, size.split("x")))
        if lora_path is not None:
            adapter_name = os.path.basename(lora_path)
            print(f"Loaded LORA weights from {adapter_name}")
            self._models[model].load_lora_weights(lora_path)
        else:
            print("Unloading LORA weights")
            self._models[model].unload_lora_weights()

        image_np = self._models[model](
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            output_type="np",
        ).images
        assert image_np.ndim == 4, f"Expected 4D array, got {image_np.ndim}D array"

        image_lst = self.postprocess(image_np)
        return image_lst

    def is_valid_model(self, model: str) -> bool:
        return model in self._models

    def postprocess(self, image_np: np.ndarray) -> list[np.ndarray]:
        image_np = (image_np * 255).round().astype("uint8")
        image_lst = np.split(image_np, image_np.shape[0], axis=0)
        image_lst = [img.squeeze(0) for img in image_lst]
        return image_lst
