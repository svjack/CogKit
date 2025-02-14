import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationInfo, field_validator
from typing_extensions import override

from ...base.base_args import BaseArgs


class DiffusionArgs(BaseArgs):
    ########## Model ##########
    model_type: Literal["i2v", "t2v", "t2i"]

    ########## Output ##########
    tracker_name: str = "diffusion-tracker"

    ########## Data Path ###########
    caption_column: Path
    image_column: Path | None = None
    video_column: Path

    ########## Training #########
    # For cogview models, train_resolution is a tuple of (height, width)
    # For cogvideo models, train_resolution is a tuple of (frames, height, width)
    train_resolution: tuple[int, int] | tuple[int, int, int]

    enable_slicing: bool = True
    enable_tiling: bool = True

    ########## Validation ##########
    validation_prompts: str | None  # if set do_validation, should not be None
    validation_images: str | None  # if set do_validation and model_type == i2v, should not be None
    validation_videos: str | None  # if set do_validation and model_type == v2v, should not be None
    gen_fps: int = 15

    @field_validator("image_column")
    def validate_image_column(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("model_type") == "i2v" and not v:
            logging.warning(
                "No `image_column` specified for i2v model. Will automatically extract first frames from videos as conditioning images."
            )
        return v

    @field_validator("validation_prompts")
    def validate_validation_required_fields(cls, v: Any, info: ValidationInfo) -> Any:
        values = info.data
        if values.get("do_validation") and not v:
            field_name = info.field_name
            raise ValueError(f"{field_name} must be specified when do_validation is True")
        return v

    @field_validator("validation_images")
    def validate_validation_images(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("do_validation") and values.get("model_type") == "i2v" and not v:
            raise ValueError(
                "validation_images must be specified when do_validation is True and model_type is i2v"
            )
        return v

    @field_validator("validation_videos")
    def validate_validation_videos(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("do_validation") and values.get("model_type") == "v2v" and not v:
            raise ValueError(
                "validation_videos must be specified when do_validation is True and model_type is v2v"
            )
        return v

    @field_validator("train_resolution")
    def validate_train_resolution(
        cls, v: tuple[int, int] | tuple[int, int, int], info: ValidationInfo
    ) -> str:
        if len(v) == 2:  # cogview models
            height, width = v

        elif len(v) == 3:  # cogvideo models
            frames, height, width = v
            # Check if (frames - 1) is multiple of 8
            if (frames - 1) % 8 != 0:
                raise ValueError("Number of frames - 1 must be a multiple of 8")

            # Check resolution for cogvideox-5b models
            model_name = info.data.get("model_name", "")
            if model_name in ["cogvideox-5b-i2v", "cogvideox-5b-t2v"]:
                if (height, width) != (480, 720):
                    raise ValueError(
                        "For cogvideox-5b models, height must be 480 and width must be 720"
                    )
        else:
            raise ValueError(
                "train_resolution must be a tuple of (height, width) for cogview models or (frames, height, width) for cogvideo models"
            )

        return v

    @override
    @classmethod
    def parse_args(cls):
        parser = cls.get_base_parser()

        # Required arguments
        parser.add_argument("--model_type", type=str, required=True)
        parser.add_argument("--caption_column", type=str, required=True)
        parser.add_argument("--video_column", type=str, required=True)
        parser.add_argument("--train_resolution", type=str, required=True)

        # Data loading
        parser.add_argument("--image_column", type=str, default=None)

        # Model configuration
        parser.add_argument("--enable_slicing", type=bool, default=True)
        parser.add_argument("--enable_tiling", type=bool, default=True)

        # Validation
        parser.add_argument("--validation_dir", type=str, default=None)
        parser.add_argument("--validation_prompts", type=str, default=None)
        parser.add_argument("--validation_images", type=str, default=None)
        parser.add_argument("--validation_videos", type=str, default=None)
        parser.add_argument("--gen_fps", type=int, default=15)

        args = parser.parse_args()

        # Convert video_resolution_buckets string to list of tuples
        frames, height, width = args.train_resolution.split("x")
        args.train_resolution = (int(frames), int(height), int(width))

        return cls(**vars(args))
