# Quick Start

## Environment Setup

Please refer to the [installation](../02-Installation.md) guide to setup your environment

<!-- TODO: clone the repo to finetune? -->

## Data Preparation

Before fine-tuning, you need to prepare your dataset according to the expected format. See the [data format](./03-Data%20Format.md) documentation for details on how to structure your data

<!-- TODO: add link to data format-->

## Start Training

:::note
Before starting training, please make sure you have read the corresponding [model card](../05-Model%20Card.md) to follow the parameter settings requirements and fine-tuning best practices
:::

<!-- TODO: move training script to cli folder? -->
<!-- TODO: add link to corresponding folder -->
1. Navigate to the `src/cogmodels/finetune/diffusion` directory

<!-- TODO: add link to training script folder -->
<!-- TODO: add link to train_ddp_t2i.sh -->
2. Choose the appropriate training script from the `scripts` directory based on your task type and distribution strategy. For example, `train_ddp_t2i.sh` corresponds to DDP strategy + text-to-image task

3. Review and adjust the parameters in the selected training script (e.g., `--data_root`, `--output_dir`, etc.)

<!-- TODO: add link to accelerate config -->
4. If you are using ZeRO strategy, refer to `accelerate_config.yaml` to confirm your ZeRO level and number of GPUs

5. Run the script, for example:
   ```bash
   bash scripts/train_ddp_t2i.sh
   ```


## Load Fine-tuned Model

### Load from LoRA

### Load from ZeRO

<!--  TODO: lora微调后如何重新加载？ SFT zero微调后如何加载？-->

<!-- TODO: 缺一个合并zero权重的脚本（合并后只有一个transformer的权重，让用户自己把这个权重
    替换到pipeline文件里，还是cli/api里提供一个transformer的权重路径？） -->
