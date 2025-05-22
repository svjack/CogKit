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

python CogKit/tools/converters/merge.py --checkpoint_dir KAEDEHARA-KAZUHA-ckpt/checkpoint-1250 --output_dir KAEDEHARA-KAZUHA-1250-merged --lora

python CogKit/tools/converters/merge.py --checkpoint_dir CogKit/gradio/lora_checkpoints/t2i/KAEDEHARA-KAZUHA-ckpt/checkpoint-1300 --output_dir KAEDEHARA-KAZUHA-1300-merged --lora

python CogKit/tools/converters/merge.py --checkpoint_dir CogKit/gradio/lora_checkpoints/t2i/KAEDEHARA-KAZUHA-ckpt/checkpoint-1350 --output_dir KAEDEHARA-KAZUHA-1350-merged --lora

#### Installation 
```bash 
pip install opencv-python-headless
pip install "cogkit@git+https://github.com/THUDM/cogkit.git"
pip install -U peft 
cogkit --help
```

import torch
from PIL import Image

from cogkit import (
    load_pipeline,
    load_lora_checkpoint,
    unload_lora_checkpoint,
    generate_image,
)

model_id_or_path = "THUDM/CogView4-6B"  # t2i generation task, for example.
pipeline = load_pipeline(
    model_id_or_path,
    transformer_path=None,
    dtype=torch.bfloat16,
)

lora_model_id_or_path = "KAEDEHARA-KAZUHA-1250-merged"
load_lora_checkpoint(pipeline, lora_model_id_or_path)

prompt = '''
这是一张现代电影风格的图片，描绘了一位年轻的男性角色。他有着白色的长发，用红色发带随意束起，在风中狂舞。猩红的瞳孔中黑色纹路流转，在月光下泛着危险的光芒。  
他站在废弃工厂的顶楼，破碎的玻璃窗在狂风中震颤。
'''
batched_image = generate_image(
    prompt=prompt,
    pipeline=pipeline,
    width=1920,
    height=1088,
    output_type="pil",
)
batched_image[0].save("output3.png")

prompt = '''
这是一张动漫风格的图片，描绘了一位年轻的男性角色。他有着白色的头发，头发上用红色的发带扎起，显得非常可爱。他的瞳孔是红色的，带有黑色的瞳孔纹样，给人一种神秘的感觉。
他独自漫步在枫叶林中，脚下是铺满落叶的小径，每一步都踩出沙沙的轻响。  
白色的发丝在微风中轻轻飘动，红色的发带像一抹跳动的火焰，与周围金红交织的枫叶相映成趣。他的红色瞳孔微微闪烁，黑色的纹样在光线下若隐若现，仿佛藏着某种不为人知的秘密。  
秋日的阳光透过枝叶的缝隙洒落，斑驳的光影在他身上流转，为他镀上一层温柔的暖色。偶尔有枫叶打着旋儿落下，擦过他的肩头，又悄然坠地。他伸手接住一片飘落的红叶，指尖轻抚过叶脉，嘴角勾起一丝若有若无的笑意。  
远处的山峦被秋色染透，层林尽染，而他只是静静地走着，仿佛与这片燃烧的枫林融为一体——既像过客，又像归人。
'''
batched_image = generate_image(
    prompt=prompt,
    pipeline=pipeline,
    width=1920,
    height=1088,
    output_type="pil",
)
batched_image[0].save("output.png")

prompt = '''
这是一张动漫风格的图片，描绘了一位年轻的男性角色。他有着白色的头发，头发上用红色的发带扎起，显得非常可爱。他的瞳孔是红色的，带有黑色的瞳孔纹样，给人一种神秘的感觉。
他站在一片辽阔的高原之上，脚下是无边无际的绿野，风拂过时掀起层层草浪，如同流动的翡翠海洋。远处，雪山巍峨耸立，峰顶终年不化的积雪在阳光下泛着冷冽的银光，与湛蓝的天空形成鲜明的对比。  
他的白色长发被高原的风轻轻扬起，红色发带像是一抹燃烧的火焰，在纯净的自然画卷中格外夺目。红色的瞳孔倒映着天地间的壮丽，黑色的纹样在光线流转间显得深邃而神秘，仿佛能看透这片土地千年的故事。  
一条蜿蜒的河流从草原间穿过，河水清澈见底，在阳光下闪烁着细碎的银光，如同大地的血脉。几只野生的羚羊在远处悠闲地低头吃草，偶尔警觉地抬头，又很快沉浸在这片宁静之中。  
夕阳西下时，天边的云霞被染成金红与紫罗兰的渐变，整片高原仿佛被镀上了一层梦幻的色彩。他静静地站在风中，衣袂翻飞，与自然融为一体——既像是一位孤独的旅人，又像是这片土地永恒的守望者。
'''
batched_image = generate_image(
    prompt=prompt,
    pipeline=pipeline,
    width=1920,
    height=1088,
    output_type="pil",
)
batched_image[0].save("output.png")
