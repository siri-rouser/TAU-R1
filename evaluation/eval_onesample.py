from pyexpat.errors import messages
import evaluate
from transformers import AutoConfig, Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText, Qwen3VLMoeForConditionalGeneration
import torch
import argparse
from pathlib import Path
from peft import PeftModel
import json
from QA_pair_database import QA_database

from peft import AutoPeftModelForCausalLM
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import time
from eval_gpt import gpt_text_eval


QAB = QA_database()

def inference(messages, model, processor):
    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )

    # split the videos and according metadatas
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    # since qwen-vl-utils has resize the images/videos, \
    # we should pass do_resize=False to avoid duplicate operation in processor!
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return output_text

def eval_main(model, processor, data_path, model_size, question, sys_prompt):
    messages = []

    if sys_prompt is not None:
        messages.append(
        {
            "role": "system",
            "content": [{"type": "text", "text": sys_prompt}],
        })  

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(data_path),
                },
                {"type": "text", "text": question},
            ],

        })
    
    output_text = inference(messages, model, processor)
    print(f"video_path: {data_path}")
    print(f"question: {question}")
    print(f"output_text: {output_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen3VL Model")
    parser.add_argument("--test-video-path", type=Path, default=None, help="Path to the evaluation video")
    parser.add_argument("--sys-prompt", type=str, default=None, help="System prompt for the evaluation")
    parser.add_argument("--question", type=str, default="What is the direct consequence of the head-on collision between the black sedan and the white sedan in the center of the intersection?", help="Question for the evaluation")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Name or path of the base VLM model")
    parser.add_argument("--model-root", type=Path, default=None, help="Path to the adapter root directory")
    args = parser.parse_args()

    base_model_id = args.base_model
    test_video_path = args.test_video_path
    model_size = base_model_id.split("-")[-2]  # e.g., "8B"

    if args.model_root is None:
        model_dir = base_model_id
    else:
        model_dir = str(args.model_root)
    print(f"Loading full fine-tuned model from: {model_dir}")
       
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    print("DEBUG: Full fine-tuned model loaded.")

    try:
        processor = AutoProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
        print("DEBUG: Processor loaded from model directory.")
    except OSError as e:
        print(f"WARNING: Processor files missing in {model_dir}. Falling back to base model processor: {base_model_id}")
        processor = AutoProcessor.from_pretrained(base_model_id, fix_mistral_regex=True)
        print("DEBUG: Processor loaded from base model.")

    eval_main(model=model, processor=processor, data_path=args.test_video_path, model_size=model_size, question=args.question, sys_prompt=args.sys_prompt)