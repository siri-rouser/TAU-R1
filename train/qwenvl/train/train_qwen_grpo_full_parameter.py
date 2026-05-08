import os
import logging
import pathlib
import sys
from pathlib import Path
from peft import PeftModel

import torch
import torch.distributed as dist
import transformers
from datasets import load_dataset

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration

from trl import GRPOConfig, GRPOTrainer
from grpo_trainner import QwenGRPOTrainer

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class  # your patch
from qwenvl.train.argument import ModelArguments, DataArguments, GRPOTrainingArguments  # reuse these if you want
from qwenvl.data.grpo_dataset import make_grpo_data_module
from grpo_utils import traffic_reward_func,category_reward_func

local_rank = None

def is_global_rank_0():
    return int(os.environ.get("RANK", 0)) == 0


def rank0_print(*args, **kwargs):
    if is_global_rank_0():
        print(*args, **kwargs)

def count_params_zerocompat(m):
    total = 0
    trainable = 0
    for p in m.parameters():
        n = getattr(p, "ds_numel", None)
        if n is None:
            # fallback: normal params
            n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = (100.0 * trainable / total) if total > 0 else 0.0
    return trainable, total, pct


def train_grpo(attn_implementation="flash_attention_2"):

    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, GRPOTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # ---- Load model ----
    if training_args.adaptor_dir:
        print(f"checkpoint found, loading model from {training_args.adaptor_dir}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            training_args.adaptor_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.temperature = float(getattr(training_args, "temperature", 1.0))
        model.generation_config.top_p = float(getattr(training_args, "top_p", 1.0))
        model.generation_config.do_sample = True

    # Set requires_grad according to args (robust: reset then enable)
    for _, p in model.named_parameters():
        p.requires_grad = False

    if model_args.tune_mm_vision:
        for _, p in model.visual.named_parameters():
            p.requires_grad = True

    if model_args.tune_mm_mlp:
        for _, p in model.visual.merger.named_parameters():
            p.requires_grad = True

    if model_args.tune_mm_llm:
        for name, p in model.named_parameters():
            # enable everything outside vision tower
            if not name.startswith("visual."):
                p.requires_grad = True

    trainable, total, pct = count_params_zerocompat(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    # ---- Processor (IMPORTANT: left padding) ----
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
        padding_side="left",  # important for GRPO :contentReference[oaicite:10]{index=10}
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # If you rely on your attention patch for packing/flattening
    if getattr(data_args, "data_flatten", False) or getattr(data_args, "data_packing", False):
        replace_qwen2_vl_attention_class()

    # ---- Build GRPO dataset ----
    data_module = make_grpo_data_module(model_args.model_name_or_path, processor, data_args)
    train_dataset = data_module["train_dataset"]

    # ---- Create GRPO trainer ----
    trainer = QwenGRPOTrainer(
        model=model,
        processing_class=processor,            # can be ProcessorMixin for VLMs :contentReference[oaicite:11]{index=11}
        reward_funcs=[category_reward_func],    
        args=training_args,
        train_dataset=train_dataset,
    )

    # ---- Train ----
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # ---- Save ----
    trainer.save_model(training_args.output_dir)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    train_grpo(attn_implementation="flash_attention_2")
