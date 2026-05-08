#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

export TORCHDYNAMO_DISABLE=1
export DISABLE_TORCH_COMPILE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MODEL_SEQ_LEN=4800 # max length of visual token
export FPS_MAX_FRAMES=90
export VIDEO_MAX_TOKEN_NUM=256

# DeepSpeed configuration
deepspeed=./scripts/zero3_offload.json

# Model configuration
llm=Qwen/Qwen3-VL-2B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=1e-6 # the suggested learning rate is from 1e-6 to 2e-7, because the norm is very high, for this stage we use a smaller learning rate than the full parameter tuning stage
batch_size=1
grad_accum_steps=8
num_generations=4

generation_batch_size=$((batch_size * NPROC_PER_NODE * grad_accum_steps))

# Training entry point
entry_file=qwenvl/train/train_qwen_grpo_full_parameter.py

# Dataset configuration
datasets="${GRPO_DATASET:-/data/Roundabout-TAU/roundabout_vau_category_qa_pairs_train.json}"

video_path="${GRPO_VIDEO_PATH:-/data}"

# Output configuration
run_name="${GRPO_RUN_NAME:-qwen3vl-2b-grpo-post-training}"
output_dir="${OUTPUT_DIR:-/output/sft_qwen3_2b_full_parameter_grpo}"
adaptor_dir="${ADAPTOR_DIR:-""}"

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --video_path ${video_path} \
    --adaptor_dir ${adaptor_dir} \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --warmup_steps 25 \
    --save_strategy "steps" \
    --save_steps 38 \
    --save_total_limit 10 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length ${MODEL_MAX_LENGTH:-16384} \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING:-True} \
    --video_max_pixels ${VIDEO_MAX_PIXELS:-$((256*28*28))} \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb \
    --num_generations ${num_generations} \
    --generation_batch_size ${generation_batch_size} \
    --max_prompt_length ${MAX_PROMPT_LENGTH:-16384} \
    --temperature 1 \
    --top_p 0.9 \
    --max_completion_length 2"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
