import hashlib
import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from safetensors.torch import load_file, save_file

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")


##########  loaders  ##########


def load_prompts(prompt_path: Path) -> list[str]:
    with open(prompt_path, encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_videos(video_path: Path) -> list[Path]:
    with open(video_path, encoding="utf-8") as file:
        return [
            video_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0
        ]


def load_images(image_path: Path) -> list[Path]:
    with open(image_path, encoding="utf-8") as file:
        return [
            image_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0
        ]


def load_images_from_videos(videos_path: list[Path]) -> list[Path]:
    first_frames_dir = videos_path[0].parent.parent / "first_frames"
    first_frames_dir.mkdir(exist_ok=True)

    first_frame_paths = []
    for video_path in videos_path:
        frame_path = first_frames_dir / f"{video_path.stem}.png"
        if frame_path.exists():
            first_frame_paths.append(frame_path)
            continue

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read video: {video_path}")

        # Save frame as PNG with same name as video
        cv2.imwrite(str(frame_path), frame)
        logging.info(f"Saved first frame to {frame_path}")

        # Release video capture
        cap.release()

        first_frame_paths.append(frame_path)

    return first_frame_paths


##########  preprocessors  ##########


def preprocess_image_with_resize(
    image: Image.Image,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Loads and resizes a single image.

    Args:
        image: PIL.Image.Image object
        height: Target height for resizing.
        width: Target width for resizing.
        device: Device to load the data on

    Returns:
        torch.Tensor: Image tensor with shape [C, H, W] where:
            C = number of channels (3 for RGB)
            H = height
            W = width
    """
    image = image.convert("RGB")
    transform = transforms.ToTensor()
    image = transform(image).float().contiguous()
    image = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    assert image.shape == (3, height, width)
    return image


def preprocess_video_with_resize(
    video: decord.VideoReader,
    max_num_frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Loads and resizes a single video.

    The function processes the video through these steps:
      1. If video frame count > max_num_frames, downsample frames evenly
      2. If video dimensions don't match (height, width), resize frames

    Args:
        video: decord.VideoReader object
        max_num_frames: Maximum number of frames to keep.
        height: Target height for resizing.
        width: Target width for resizing.
        device: Device to load the data on

    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    video_num_frames = len(video)
    if video_num_frames < max_num_frames:
        # Get all frames first
        frames = video.get_batch(list(range(video_num_frames))).float().to(device)

        # Resize frames to target dimensions
        frames = frames.permute(0, 3, 1, 2)  # [F, H, W, C] -> [F, C, H, W]
        frames = torch.nn.functional.interpolate(
            frames,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        # Repeat the last frame until we reach max_num_frames
        last_frame = frames[-1:]
        num_repeats = max_num_frames - video_num_frames
        repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)
        frames = torch.cat([frames, repeated_frames], dim=0)

    else:
        indices = list(range(0, video_num_frames, video_num_frames // max_num_frames))
        frames = video.get_batch(indices).float().to(device)

        # Resize frames to target dimensions
        frames = frames.permute(0, 3, 1, 2)  # [F, H, W, C] -> [F, C, H, W]
        frames = torch.nn.functional.interpolate(
            frames,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        frames = frames[:max_num_frames]

    return frames.contiguous()


##########  embedding & caching  ##########


def get_prompt_embedding(
    encode_fn: Callable, prompt: str, cache_dir: Path, logger: logging.Logger
) -> torch.Tensor:
    """Get prompt embedding from cache or create new one if not exists.

    Args:
        encode_fn: Function to project prompt to embedding.
        prompt: Text prompt to be embedded
        cache_dir: Base directory for caching embeddings
        logger: Logger instance for logging messages

    Returns:
        torch.Tensor: Prompt embedding with shape [seq_len, hidden_size]
    """
    prompt_embeddings_dir = cache_dir / "prompt_embeddings"
    prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

    prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
    prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")

    if prompt_embedding_path.exists():
        prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
        logger.debug(
            f"Loaded prompt embedding from {prompt_embedding_path}",
            main_process_only=False,
        )
    else:
        prompt_embedding = encode_fn(prompt)
        prompt_embedding = prompt_embedding.to("cpu")
        # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
        prompt_embedding = prompt_embedding[0]
        save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
        logger.info(
            f"Saved prompt embedding to {prompt_embedding_path}",
            main_process_only=False,
        )

    return prompt_embedding
