git clone https://huggingface.co/datasets/svjack/Nino_Videos_Captioned

import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import shutil

def change_resolution_and_save(input_path, output_path, target_width=1024, target_height=768, max_duration=4):
    """Process images and videos to target resolution and split videos into segments."""
    os.makedirs(output_path, exist_ok=True)

    for root, dirs, files in os.walk(input_path):
        for file in tqdm(files, desc="Processing files"):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, input_path)
            output_dir = os.path.dirname(os.path.join(output_path, relative_path))

            # Process images
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = cv2.imread(file_path)
                    h, w = img.shape[:2]
                    scale = min(target_width / w, target_height / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                    x_offset = (target_width - new_w) // 2
                    y_offset = (target_height - new_h) // 2
                    background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
                    output_file_path = os.path.join(output_path, relative_path)
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    cv2.imwrite(output_file_path, background)

                    # Copy corresponding txt file
                    base_name = os.path.splitext(file)[0]
                    txt_source = os.path.join(root, f"{base_name}.txt")
                    if os.path.exists(txt_source):
                        txt_target = os.path.join(output_dir, f"{base_name}.txt")
                        shutil.copy2(txt_source, txt_target)
                except Exception as e:
                    print(f"Failed to process image {file_path}: {e}")

            # Process videos
            elif file.lower().endswith('.mp4'):
                try:
                    clip = VideoFileClip(file_path)
                    total_duration = clip.duration
                    base_name = os.path.splitext(file)[0]

                    if total_duration <= max_duration:
                        # Process the entire video
                        output_filename = f"{base_name}.mp4"
                        output_file_path = os.path.join(output_dir, output_filename)
                        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                        def process_frame(frame):
                            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            h, w = img.shape[:2]
                            scale = min(target_width / w, target_height / h)
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                            background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                            x_offset = (target_width - new_w) // 2
                            y_offset = (target_height - new_h) // 2
                            background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
                            return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

                        processed_clip = clip.fl_image(process_frame)
                        fps = processed_clip.fps if processed_clip.fps else 24
                        processed_clip.write_videofile(
                            output_file_path,
                            codec='libx264',
                            fps=fps,
                            preset='slow',
                            threads=4,
                            audio=False
                        )
                        processed_clip.close()

                        # Copy corresponding txt file
                        txt_source = os.path.join(root, f"{base_name}.txt")
                        if os.path.exists(txt_source):
                            txt_target = os.path.join(output_dir, f"{base_name}.txt")
                            shutil.copy2(txt_source, txt_target)
                    else:
                        # Split and process the video
                        num_segments = int(total_duration // max_duration)
                        for i in range(num_segments):
                            start_time = i * max_duration
                            end_time = min((i+1) * max_duration, total_duration)
                            sub_clip = clip.subclip(start_time, end_time)

                            output_filename = f"{base_name}_{i}.mp4"
                            output_file_path = os.path.join(output_dir, output_filename)
                            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                            def process_frame(frame):
                                img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                h, w = img.shape[:2]
                                scale = min(target_width / w, target_height / h)
                                new_w = int(w * scale)
                                new_h = int(h * scale)
                                resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                                background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                                x_offset = (target_width - new_w) // 2
                                y_offset = (target_height - new_h) // 2
                                background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
                                return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

                            processed_clip = sub_clip.fl_image(process_frame)
                            fps = processed_clip.fps if processed_clip.fps else 24
                            processed_clip.write_videofile(
                                output_file_path,
                                codec='libx264',
                                fps=fps,
                                preset='slow',
                                threads=4,
                                audio=False
                            )
                            processed_clip.close()

                            # Copy corresponding txt file
                            txt_source = os.path.join(root, f"{base_name}.txt")
                            if os.path.exists(txt_source):
                                txt_target = os.path.join(output_dir, f"{base_name}_{i}.txt")
                                shutil.copy2(txt_source, txt_target)

                    clip.close()
                except Exception as e:
                    print(f"Failed to process video {file_path}: {e}")

# Example usage
change_resolution_and_save(
    input_path="Nino_Videos_Captioned",
    output_path="Nino_Videos_Captioned_720x480x6",
    target_width=720,
    target_height=480,
    max_duration=6
)

import json
import os
import shutil
from pathlib import Path
import random

# 设置路径和参数
input_dir = "Nino_Videos_Captioned_720x480x6"
output_dir = "Nino_Videos_Captioned_720x480x6_processed"
test_size = 0.05  # 20%的数据作为测试集

# 创建输出目录结构
os.makedirs(os.path.join(output_dir, "train", "videos"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test", "videos"), exist_ok=True)

# 获取所有文件对
files = []
for file in os.listdir(input_dir):
    if file.endswith(".mp4"):
        base_name = os.path.splitext(file)[0]
        txt_file = base_name + ".txt"
        if os.path.exists(os.path.join(input_dir, txt_file)):
            files.append((file, txt_file))

# 随机打乱文件顺序
random.shuffle(files)

# 分割训练集和测试集
split_idx = int(len(files) * (1 - test_size))
train_files = files[:split_idx]
test_files = files[split_idx:]

# 处理训练集
with open(os.path.join(output_dir, "train", "metadata.jsonl"), "w") as f:
    for idx, (mp4_file, txt_file) in enumerate(train_files):
        # 读取文本内容
        with open(os.path.join(input_dir, txt_file), "r") as txt_f:
            caption = txt_f.read().strip()

        # 生成唯一文件名
        video_path = f"videos/{idx:05d}.mp4"

        # 复制视频文件
        shutil.copy2(
            os.path.join(input_dir, mp4_file),
            os.path.join(output_dir, "train", video_path.split("/")[-1])
        )

        # 写入元数据
        metadata = {
            "file_name": video_path.split("/")[-1],
            "prompt": caption
        }
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

# 处理测试集
with open(os.path.join(output_dir, "test", "prompt.jsonl"), "w") as f:
    for idx, (mp4_file, txt_file) in list(enumerate(test_files))[:2]:
        # 读取文本内容
        with open(os.path.join(input_dir, txt_file), "r") as txt_f:
            caption = txt_f.read().strip()

        # 写入元数据
        metadata = {
            "prompt": caption
        }
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

vim cogvideox_2b.yaml

```yaml
# ================ Logging ================
name4train: "t2v-train"
log_level: "INFO"  # Options: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# ================ Model ================
model_name: "cogvideox-t2v"  # Options: ["cogvideox-t2v", "cogvideox1.5-t2v"]
model_path: "THUDM/CogVideoX-2B"

# ================ Output ================
output_dir: "Nino_Lora_output"


# ================ Tracker ================
report_to: null  # Options: ["wandb"]


# ================ Data ================
data_root: "../Nino_Videos_Captioned_720x480x6_processed"


# ================ Training ================
seed: 42
training_type: "lora"   # Options: ["lora", "sft"]

strategy: "DDP"  # Options: ["DDP", "SHARD_GRAD_OP", "FULL_SHARD", "HYBRID_SHARD", "_HYBRID_SHARD_ZERO2"]

# This will offload model param and grads to CPU memory to save GPU memory, but will slow down training
offload_params_grads: false

# This will increase memory usage since gradients are sharded during accumulation step.
# Note: When used with offload_params_grads, model parameters and gradients will only be offloaded
#   to the CPU during the final synchronization (still retained on GPU in gradient accumulation steps)
#   which means offload_params_grads is meaningless when used with no_grad_sync_when_accumulating
no_grad_sync_when_accumulating: false

# When enable_packing is true, training will use the native image resolution,
#   otherwise all images will be resized to train_resolution, which may distort the original aspect ratio.
# IMPORTANT: When changing enable_packing from true to false (or false to true),
#   make sure to clear the `.cache` directories in your `data_root/train` and `data_root/test` folders if they exist.
enable_packing: false

# Note:
#   for CogVideoX series models, number of training frames should be **8N+1**
#   for CogVideoX1.5 series models, number of training frames should be **16N+1**
train_resolution: [81, 480, 720]  # [Frames, Height, Width]

train_epochs: 500
batch_size: 1
gradient_accumulation_steps: 1
mixed_precision: "bf16"  # Options: ["fp32", "fp16", "bf16"]
learning_rate: 2.0e-5

num_workers: 8
pin_memory: true

checkpointing_steps: 50
checkpointing_limit: 20
resume_from_checkpoint: null  # or "/path/to/checkpoint/dir"


# ================ Validation ================
do_validation: true
validation_steps: 50  # Must be a multiple of `checkpointing_steps`
gen_fps: 16
```

torchrun \
    --nproc_per_node=1 \
    --master_port=29501 \
    quickstart/scripts/train.py \
    --yaml cogvideox_2b.yaml
