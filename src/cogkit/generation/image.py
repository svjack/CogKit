# -*- coding: utf-8 -*-


import os
from functools import partial
from pathlib import Path

import torch
from diffusers import CogView4ControlPipeline, CogView4Pipeline
from PIL import Image

from cogkit.generation.util import before_generation, guess_resolution
from cogkit.logging import get_logger
from cogkit.types import GenerationMode
from cogkit.utils import convert_prompt, mkdir, rand_generator, resolve_path

_logger = get_logger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_image(
    task: GenerationMode,
    prompt: str,
    model_id_or_path: str,
    output_file: str | Path,
    image_file: str | Path | None = None,
    # * params for model loading
    dtype: torch.dtype = torch.bfloat16,
    transformer_path: str | None = None,
    # * params for generated images
    height: int | None = None,
    width: int | None = None,
    # * params for the generation process
    num_inference_steps: int = 50,
    guidance_scale: float = 3.5,
    seed: int | None = 42,
):
    pipeline = None
    if task == GenerationMode.TextToImage:
        pipeline = CogView4Pipeline.from_pretrained(model_id_or_path, torch_dtype=dtype).to(device)
    elif task == GenerationMode.CtrlTextToImage:
        pipeline = CogView4ControlPipeline.from_pretrained(model_id_or_path, torch_dtype=dtype).to(
            device
        )
    else:
        err_msg = f"Unknown generation mode: {task.value}"
        raise ValueError(err_msg)

    if transformer_path is not None:
        pipeline.transformer.save_config(transformer_path)
        pipeline.transformer = pipeline.transformer.from_pretrained(transformer_path)

    height, width = guess_resolution(pipeline, height, width)

    _logger.info(f"Generation config: height {height}, width {width}.")

    before_generation(pipeline)
    pipeline_fn = partial(
        pipeline,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        generator=rand_generator(seed),
    )
    enhanced_prompt = convert_prompt(prompt, task, retry_times=5)

    if task == GenerationMode.TextToImage:
        batch_image = pipeline_fn().images
    elif task == GenerationMode.CtrlTextToImage:
        batch_image = pipeline_fn(
            prompt=enhanced_prompt,
            control_image=Image.open(image_file),
        ).images
    else:
        err_msg = f"Unknown generation mode: {task.value}"
        raise ValueError(err_msg)

    output_file = resolve_path(output_file)
    mkdir(output_file.parent)
    _logger.info("Saving the generated image to path '%s'.", os.fspath(output_file))
    batch_image[0].save(output_file)
