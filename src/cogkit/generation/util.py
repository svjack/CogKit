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


def guess_resolution(
    pipeline: DiffusionPipeline,
    height: int | None = None,
    width: int | None = None,
) -> tuple[int, int]:
    pl_cls_name = pipeline.__class__.__name__
    if pl_cls_name.startswith("CogView"):
        return _guess_cogview_resolution(pipeline, height=height, width=width)
    if pl_cls_name.startswith("CogVideoX"):
        return _guess_cogvideox_resolution(pipeline, height=height, width=width)

    err_msg = f"The pipeline '{pl_cls_name}' is not supported."
    raise ValueError(err_msg)


def _guess_cogview_resolution(
    pipeline: DiffusionPipeline,
    height: int | None = None,
    width: int | None = None,
) -> tuple[int, int]:
    # TODO: completes this
    raise NotImplementedError


def _guess_cogvideox_resolution(
    pipeline: DiffusionPipeline, height: int | None, width: int | None = None
) -> tuple[int, int]:
    # TODO: completes this
    raise NotImplementedError


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
