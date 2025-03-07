# -*- coding: utf-8 -*-


import os
from pathlib import Path

from cogmodels.logging import get_logger
from cogmodels.types import GenerationMode
from cogmodels.utils.diffusion_pipeline import get_pipeline_meta
from cogmodels.utils.path import resolve_path

_logger = get_logger(__name__)


def guess_generation_mode(
    model_id_or_path: str,
    generation_mode: str | GenerationMode | None = None,
    image_file: str | Path | None = None,
    video_file: str | Path | None = None,
) -> GenerationMode:
    pipeline_meta = get_pipeline_meta(model_id_or_path)
    pl_cls_name = pipeline_meta.get("cls_name", None)
    if pl_cls_name is None:
        err_msg = f"Failed to parse the pipeline configuration (pipeline_cls = {pl_cls_name})."
        raise ValueError(err_msg)

    if pl_cls_name.startswith("CogView"):
        return GenerationMode.TextToImage
    # FIXME: check if I2V pipeline
    if not pl_cls_name.startswith("CogVideo"):
        err_msg = f"Unknown diffusion pipeline: {pl_cls_name}"
        raise ValueError(err_msg)

    if generation_mode is not None:
        return GenerationMode(generation_mode)

    def _validate_file(path: str | Path | None) -> Path | None:
        if path is None:
            return None

        path = resolve_path(path)
        if not path.is_file():
            _logger.warning(
                "Path '%s' is not a regular file. Will ignore it.",
                os.fspath(path),
            )
            return None
        return path

    valid_img_file = _validate_file(image_file)
    valid_vid_file = _validate_file(video_file)

    if valid_img_file is not None and valid_vid_file is not None:
        _logger.warning(
            "Both image and video input are received. Will ignore the video."
        )
        valid_vid_file = None

    if valid_img_file is not None:
        return GenerationMode.ImageToVideo
    if valid_vid_file is not None:
        return GenerationMode.VideoToVideo
    return GenerationMode.TextToVideo
