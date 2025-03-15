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

    Note: The value of returned image tensor should be in the range of 0 ~ 255(rather than 0 ~ 1).
    """
    image = image.convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().contiguous()

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
        assert prompt_embedding.ndim == 2
        # shape of prompt_embedding: [seq_len, hidden_size]

        prompt_embedding = prompt_embedding.to("cpu")
        save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
        logger.info(
            f"Saved prompt embedding to {prompt_embedding_path}",
            main_process_only=False,
        )

    return prompt_embedding


def get_image_embedding(
    encode_fn: Callable, image: Image.Image, cache_dir: Path, logger: logging.Logger
) -> torch.Tensor:
    """Get encoded image from cache or create new one if not exists.

    Args:
        encode_fn: Function to project image to embedding.
        image: Image to be embedded
        cache_dir: Base directory for caching embeddings
        logger: Logger instance for logging messages

    Returns:
        torch.Tensor: Encoded image with shape [C, H, W]
    """
    encoded_images_dir = cache_dir / "encoded_images"
    encoded_images_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(image.filename).stem
    filename_hash = str(hashlib.sha256(filename.encode()).hexdigest())
    encoded_image_path = encoded_images_dir / (filename_hash + ".safetensors")

    if encoded_image_path.exists():
        encoded_image = load_file(encoded_image_path)["encoded_image"]
        logger.debug(
            f"Loaded encoded image from {encoded_image_path}",
            main_process_only=False,
        )
    else:
        encoded_image = encode_fn(image)
        encoded_image = encoded_image.to("cpu")

        # shape of encoded_image: [C, H, W]
        save_file({"encoded_image": encoded_image}, encoded_image_path)
        logger.info(
            f"Saved encoded image to {encoded_image_path}",
            main_process_only=False,
        )

    return encoded_image
