#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "/ome/lhy/code/cogmodels/CogView4-6B"
    --model_name "cogview4-6b"
    --model_type "t2i"
    --training_type "sft"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "/home/lhy/code/cogmodels/src/cogmodels/finetune/diffusion/train_result/cogview4/sft-pred-noise"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    # --data_root "/home/lhy/code/cogmodels/src/cogmodels/finetune/data/t2i"
    --data_root "/home/lhy/code/cogmodels/src/cogmodels/finetune/data/t2i-foo"
    --train_resolution "1024x1024"  # (height x width)
)

# Training Configuration
TRAIN_ARGS=(
    # --train_epochs 1 # number of training epochs
    # --train_epochs 100 # number of training epochs
    --train_epochs 10000 # number of training epochs
    --seed 42 # random seed

    --learning_rate 2e-5
    # --learning_rate 0

    #########   Please keep consistent with deepspeed config file ##########
    --batch_size 2
    # --batch_size 3
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] Only CogVideoX-2B supports fp16 training
    ########################################################################

)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 250 # save checkpoint every x steps
    # --checkpointing_steps 10 # save checkpoint every x steps
    --checkpointing_limit 1 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_steps 250  # should be multiple of checkpointing_steps
    # --validation_steps 10  # should be multiple of checkpointing_steps
)

# Combine all arguments and launch training
accelerate launch --config_file ./accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
