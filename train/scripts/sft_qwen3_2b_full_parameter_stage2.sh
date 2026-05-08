#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

export MODEL_SEQ_LEN=12000 # max length of visual token
export FPS_MAX_FRAMES=180
export VIDEO_MAX_TOKEN_NUM=256

# DeepSpeed configuration
deepspeed=./scripts/zero3_offload.json

# Model configuration
stage1_output_dir="${STAGE1_OUTPUT_DIR:-/output/sft_qwen3_2b_full_parameter_stage1}"
llm=Qwen/Qwen3-VL-2B-Instruct  # Keep base model id so AutoProcessor can load reliably

# Training hyperparameters
lr=5e-6 # the suggested learning rate is from 1e-6 to 2e-7
batch_size=1
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets="roundabout_vau_category_train"

# Output configuration
run_name="qwen3vl_roundabout_vau_full_parameter_stage2"
adaptor_dir="${stage1_output_dir}"
output_dir=${OUTPUT_DIR:-/output/sft_qwen3_2b_full_parameter_stage2}

if [ ! -d "${stage1_output_dir}" ]; then
    echo "[ERROR] stage1 output directory not found: ${stage1_output_dir}"
    echo "Run stage1 first or update stage1_output_dir in this script."
    exit 1
fi

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --adaptor_dir ${adaptor_dir} \
    --num_train_epochs 8 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels $((256*28*28)) \
    --min_pixels $((32*28*28)) \
    --video_max_pixels $((256*28*28)) \
    --video_min_pixels $((32*28*28)) \
    --video_max_frames ${FPS_MAX_FRAMES} \
    --warmup_steps 25 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 76 \
    --save_total_limit 3 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb "

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}