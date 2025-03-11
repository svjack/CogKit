# CogModel

## Updates

- 2025-mm-DD, release and open source cogmodel.

## Introduction

**CogModels** is an open-source initiative by Zhipu AI that provides a user-friendly interface, enabling researchers and developers to access and manipulate the Cog family of models. The project aims to streamline the application of Cog models across multimodal generation tasks such as **text-to-image (t2i)**, **text-to-video (t2v)**, **image-to-video (i2v)**, and **video-to-video (v2v)**. It should be noted that utilization of CogModels and associated Cog models must adhere to relevant legal frameworks and ethical guidelines to ensure responsible and ethical implementation.

## Features
- Multiple models: CogVideoX, CogVideoX1.5, Cogview3, Cogview4, etc.
- Ensemble methods: (incremental) pre-training, (multimodal) instruction.
- Multiple precisions: 16-bit full parameter fine-tuning, frozen fine-tuning, LoRA fine-tuning.
- Fine-tuning methods: single machine single card, single machine multiple cards, multiple machines multiple cards.
- Wide range of tasks: multi-round dialogue, image generation, video generation, etc.
- Extreme reasoning: based on OpenAI style API, browser interface and command line interface.

## Supported Models
### CogVideo series models

repo: https://github.com/THUDM/CogVideo

|Model Name| Generation Task | dtype | Recommend Resolution<br>(height * width) | Checkpoint |
|:---:|:---:|:---:|:---:|:---|
| CogVideoX-5B-I2V | image-to-video | bfloat16<br>float16 | 720 * 480 | 🤗 [HuggingFace](https://huggingface.co/THUDM/CogVideoX-5b-I2V) <br> 🤖 [ModelScope](https://modelscope.cn/models/ZhipuAI/CogVideoX-5b-I2V) <br> 🟣 [WiseModel](https://wisemodel.cn/models/ZhipuAI/CogVideoX-5b-I2V) |
| CogVideoX-5B | text-to-image <br> video-to-video | bfloat16<br>float16 | 720 * 480 | 🤗 [HuggingFace](https://huggingface.co/THUDM/CogVideoX-5b) <br> 🤖 [ModelScope](https://modelscope.cn/models/ZhipuAI/CogVideoX-5b) <br> 🟣 [WiseModel](https://wisemodel.cn/models/ZhipuAI/CogVideoX-5b) |
| CogVideoX-2B | text-to-image <br> video-to-video | bfloat16<br>float16 | 720 * 480 | 🤗 [HuggingFace](https://huggingface.co/THUDM/CogVideoX-2b) <br> 🤖 [ModelScope](https://modelscope.cn/models/ZhipuAI/CogVideoX-2b) <br> 🟣 [WiseModel](https://wisemodel.cn/models/ZhipuAI/CogVideoX-2b) |
| CogVideoX1.5-5B-I2V (Latest) | image-to-video | bfloat16<br>float16 | Min(W, H) = 768 <br> 768 ≤ Max(W, H) ≤ 1360 <br> Max(W, H) \mod 16 = 0 | 🤗 [HuggingFace](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V) <br> 🤖 [ModelScope](https://modelscope.cn/models/ZhipuAI/CogVideoX1.5-5B-I2V) <br> 🟣 [WiseModel](https://wisemodel.cn/models/ZhipuAI/CogVideoX1.5-5B-I2V) |
| CogVideoX1.5-5B (Latest) | text-to-image <br> video-to-video | bfloat16<br>float16 | 1360 * 768 | 🤗 [HuggingFace](https://huggingface.co/THUDM/CogVideoX1.5-5B) <br> 🤖 [ModelScope](https://modelscope.cn/models/ZhipuAI/CogVideoX1.5-5B) <br> 🟣 [[WiseModel](https://wisemodel.cn/models/ZhipuAI/CogVideoX1.5-5B)] |

### CogView series models

repo: https://github.com/THUDM/CogView4

|Model Name| Generation Task | dtype | Recommend Resolution<br>(height * width) | Checkpoint |
|:---:|:---:|:---:|:---:|:---|
| CogView3-Base-3B | text-to-image | bfloat16<br>float16 | 512 * 512 | Not Adapted |
| CogView3-Base-3B-distill | text-to-image | bfloat16<br>float16 | 512 * 512 |Not Adapted |
| CogView3-Plus-3B | text-to-image | bfloat16<br>float16 | 512 <= H, W <= 2048 <br> H * W <= 2^{21} <br> H, W \mod 32 = 0 | 🤗 [HuggingFace](https://huggingface.co/THUDM/CogView3-Plus-3B) <br> 🤖 [ModelScope](https://modelscope.cn/models/ZhipuAI/CogView3-Plus-3B) <br> 🟣 [WiseModel](https://wisemodel.cn/models/ZhipuAI/CogView3-Plus-3B) |
| CogView4-6B | text-to-image | bfloat16<br>float16 | 512 <= H, W <= 2048 <br> H * W <= 2^{21} <br> H, W \mod 32 = 0 | 🤗 [HuggingFace](https://huggingface.co/THUDM/CogView4-6B) <br> 🤖 [ModelScope](https://modelscope.cn/models/ZhipuAI/CogView4-6B) <br> 🟣 [WiseModel](https://wisemodel.cn/models/ZhipuAI/CogView4-6B) |

## Usage
### Installation

```bash
pip install cogmodels
```

### Inference

#### CLI

```text
Usage: python -m cogmodels inference [OPTIONS] PROMPT MODEL_ID_OR_PATH SAVE_FILE

Options:
  --task [t2v|i2v|v2v|t2i]             select the task type in t2v, i2v, v2v, t2i
  --image_file FILE                    the input image file
  --video_file FILE                    the input video file
  --lora_model_id_or_path TEXT         the id or the path of the LoRA weights
  --lora_rank INTEGER RANGE            the rank of the LoRA weights  [x>=1]
  --dtype [bfloat16|float16]           the data type used in the computation
  --num_frames INTEGER RANGE           the number of the frames in the generated video (NOT EFFECTIVE in the image generation task) [x>=1]
  --fps INTEGER RANGE                  the frames per second of the generated video (NOT EFFECTIVE in the image generation task) [x>=1]
  --num_inference_steps INTEGER RANGE  the number of the diffusion steps  [x>=1]
  --seed INTEGER                       the seed for reproducibility
  --help                               Show this message and exit.
```

#### Quick start
```bash
python -m cogmodels "a flying dog" ${PATH_TO_COGVIDEO} ${SAVE_FILE}
```

### Finetune


#### API Server

### Post Training
