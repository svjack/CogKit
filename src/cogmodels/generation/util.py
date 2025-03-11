# -*- coding: utf-8 -*-


from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
    DiffusionPipeline,
)

# Recommended resolution for each model (width, height)
# RESOLUTION_MAP = {
#     # cogvideox1.5-*
#     "cogvideox1.5-5b-i2v": (768, 1360),
#     "cogvideox1.5-5b": (768, 1360),
#     # cogvideox-*
#     "cogvideox-5b-i2v": (480, 720),
#     "cogvideox-5b": (480, 720),
#     "cogvideox-2b": (480, 720),
# }


# FIXME: whether to keep this
def guess_resolution(
    pipeline: DiffusionPipeline,
    height: int | None = None,
    width: int | None = None,
) -> tuple[int, int]:
    if height is not None and width is not None:
        return (height, width)

    height = height or pipeline.transformer.config.sample_height * pipeline.vae_scale_factor_spatial
    width = width or pipeline.transformer.config.sample_width * pipeline.vae_scale_factor_spatial
    return (height, width)


# FIXME: whether to keep this
def guess_image_resolution(
    pipeline: DiffusionPipeline,
    height: int | None = None,
    width: int | None = None,
) -> tuple[int, int]:
    if height is not None and width is not None:
        return (height, width)
    height = height or pipeline.transformer.config.sample_size * pipeline.vae_scale_factor
    width = width or pipeline.transformer.config.sample_size * pipeline.vae_scale_factor
    return (height, width)


def before_generation(pipeline: DiffusionPipeline):
    if isinstance(
        pipeline,
        CogVideoXPipeline | CogVideoXImageToVideoPipeline | CogVideoXVideoToVideoPipeline,
    ):
        pipeline.scheduler = CogVideoXDPMScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )

    # * enables CPU offload for the model.
    # turns off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")
    # pipe.to("cuda")

    # pipeline.to("cuda")
    pipeline.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    if hasattr(pipeline, "vae"):
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
