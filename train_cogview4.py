from diffusers import CogView4Pipeline
import torch

pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16)
# Open it for reduce GPU memory usage
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()


prompt = '''
远景，古代中国风景。蛙鼓闹塘：水下镜头捕捉墨绿蛙背顶破浮萍的慢动作，蝌蚪群游动轨迹在淤泥上留下梵高星空般的抽象纹路。水面倒影里，整片桃林正在风中摇曳成印象派画作。'''
image = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    num_images_per_prompt=1,
    num_inference_steps=50,
    width=1920,
    height=1072,
).images[0]

image.save("cogview4_8.png")


import json
import os
from datasets import load_dataset

# 加载数据集
ds = load_dataset("svjack/genshin_impact_KAEDEHARA_KAZUHA_Omni_Captioned")

# 创建目录结构
os.makedirs("genshin_impact_KAEDEHARA_KAZUHA_Omni_Captioned/train/images", exist_ok=True)
os.makedirs("genshin_impact_KAEDEHARA_KAZUHA_Omni_Captioned/test", exist_ok=True)

# 处理训练集
with open("genshin_impact_KAEDEHARA_KAZUHA_Omni_Captioned/train/metadata.jsonl", "w") as f:
    for idx, example in enumerate(ds["train"]):
        # 生成唯一文件名（如果原图没有filename属性）
        #image_path = f"images/{idx:05d}.png"  # 使用5位数字编号
        image_path = f"{idx:05d}.png"  # 使用5位数字编号

        # 保存图片（PIL对象直接保存）
        example["image"].save(
            os.path.join("genshin_impact_KAEDEHARA_KAZUHA_Omni_Captioned/train", image_path)
        )

        # 写入元数据
        metadata = {
            "file_name": image_path.split("/")[-1],
            "prompt": example["prompt"]
        }
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")  # 处理中文

# 处理测试集（取前3个样本作为示例）
with open("genshin_impact_KAEDEHARA_KAZUHA_Omni_Captioned/test/prompt.jsonl", "w") as f:
    for idx, example in enumerate(ds["train"]):
        if idx >= 5:
            break
        # 写入元数据
        metadata = {
            "prompt": example["prompt"]
        }
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")  # 处理中文

cogview-4b-lora.yaml
```yaml
# ================ Logging ================
name4train: "t2i-train"
log_level: "INFO"  # Options: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# ================ Model ================
model_name: "cogview4-6b"  # Options: ["cogview4-6b"]
model_path: "THUDM/CogView4-6B"
model_type: "t2i"

# ================ Output ================
output_dir: "/home/featurize/CogKit/gradio/lora_checkpoints/t2i/KAEDEHARA-KAZUHA-ckpt"

# ================ Tracker ================
#report_to: "tensorboard"  # Options: ["wandb"]
report_to: null

# ================ Data ================
data_root: "../genshin_impact_KAEDEHARA_KAZUHA_Omni_Captioned"

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

# This will slow down validation speed and enable quantization during training to save GPU memory
low_vram: false

# Note: For CogView4 series models, height and width should be **32N** (multiple of 32)
train_resolution: [1024, 1024]  # [Height, Width]

train_epochs: 500
batch_size: 1
gradient_accumulation_steps: 4
mixed_precision: "bf16"  # Options: ["fp32", "fp16", "bf16"]
learning_rate: 2.0e-5

num_workers: 8
pin_memory: true
nccl_timeout: 1800

checkpointing_steps: 50
checkpointing_limit: 20
resume_from_checkpoint: null  # or "/path/to/checkpoint/dir"

# ================ Validation ================
do_validation: true
validation_steps: 50  # Must be a multiple of `checkpointing_steps`
```

torchrun \
    --nproc_per_node=1 \
    --master_port=29501 \
    quickstart/scripts/train.py \
    --yaml cogview-4b-lora.yaml
