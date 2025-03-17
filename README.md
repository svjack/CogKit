# CogModel

## Updates

- 2025-mm-DD, release and open source cogmodel.

## Introduction

**CogModels** is an open-source initiative by Zhipu AI that provides a user-friendly interface, enabling researchers and developers to access and manipulate the Cog family of models， you can check [here](docs/05-Model%20Card.md) to view support models. The project aims to streamline the application of Cog models across multimodal generation tasks such as **text-to-image (t2i)**, **text-to-video (t2v)**, **image-to-video (i2v)**. It should be noted that utilization of CogModels and associated Cog models must adhere to relevant legal frameworks and ethical guidelines to ensure responsible and ethical implementation.

## Features

- Multiple models: CogVideoX, CogVideoX1.5, Cogview3, Cogview4, etc.
- Ensemble methods: (incremental) pre-training, (multimodal) instruction.
- Multiple precisions: 16-bit full parameter fine-tuning, frozen fine-tuning, LoRA fine-tuning.
- Fine-tuning methods: single machine single card, single machine multiple cards, multiple machines multiple cards.
- Wide range of tasks: multi-round dialogue, image generation, video generation, etc.
- Extreme reasoning: based on OpenAI style API, browser interface and command line interface.

## Usage

### Installation

```bash
pip install cogmodels
```

### Inference

#### CLI

```text
Usage: python -m cogmodels inference [OPTIONS] PROMPT MODEL_ID_OR_PATH OUTPUT_FILE

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
python -m cogmodels "a flying dog" ${PATH_TO_COGVIDEO} ${OUTPUT_FILE}
```

### Finetune

#### API Server

### Post Training
