from pathlib import Path
from typing import Any

<<<<<<< HEAD
from cogmodels.finetune.base import BaseState
=======
from cogkit.finetune.base import BaseState
>>>>>>> test/main


class DiffusionState(BaseState):
    transformer_config: dict[str, Any] = None

    # for video input, train_resolution = (frames, height, width)
    # for image input, train_resolution = (height, width)
    train_resolution: tuple[int, int, int] | tuple[int, int]

    validation_prompts: list[str] = []
    validation_images: list[Path | None] = []
    validation_videos: list[Path | None] = []
