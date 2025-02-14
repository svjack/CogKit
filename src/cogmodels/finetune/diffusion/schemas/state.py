from pathlib import Path
from typing import Any

from cogmodels.finetune.base import BaseState


class DiffusionState(BaseState):
    transformer_config: dict[str, Any] = None
    train_frames: int
    train_height: int
    train_width: int

    validation_prompts: list[str] = []
    validation_images: list[Path | None] = []
    validation_videos: list[Path | None] = []
