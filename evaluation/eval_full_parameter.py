import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import evaluate
from sklearn.metrics import accuracy_score, f1_score
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from eval_gpt import deepseek_text_eval, gemini_text_eval, claude_text_eval


def quantitative_eval(references, predictions):
    """Classification metrics used for the 'category' evaluation task."""
    print("References:", references)
    print("Predictions:", predictions)

    binary_references = [1 if ref in ["B", "C", "D"] else 0 for ref in references]
    binary_predictions = [1 if pred in ["B", "C", "D"] else 0 for pred in predictions]

    results = {
        "Accuracy": accuracy_score(references, predictions),
        "F1": f1_score(references, predictions, average="weighted"),
        "Binary Accuracy": accuracy_score(binary_references, binary_predictions),
        "Binary F1": f1_score(binary_references, binary_predictions, average="weighted"),
        "References": " | ".join(map(str, references)),
        "Predictions": " | ".join(map(str, predictions)),
    }
    return results


def text_eval(references, predictions, video_paths, output_path, category, model_size, checkpoint, max_workers=4):
    """BLEU/METEOR/ROUGE plus LLM-judge metrics used for the 'description' evaluation task."""
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    meteor_scores = meteor.compute(predictions=predictions, references=references)
    bleu_scores = bleu.compute(predictions=predictions, references=[[r] for r in references])
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    judges = {
        "deepseek": lambda: deepseek_text_eval(references, predictions, video_paths, output_path, category, model_size=model_size, checkpoint=checkpoint, max_workers=8),
        "gemini":   lambda: gemini_text_eval(references, predictions, video_paths, output_path, category, model_size=model_size, checkpoint=checkpoint, max_workers=8),
        "claude":   lambda: claude_text_eval(references, predictions, video_paths, output_path, category, model_size=model_size, checkpoint=checkpoint, max_workers=2),
    }
    scores = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn): name for name, fn in judges.items()}
        for future in as_completed(futures):
            name = futures[future]
            scores[name] = future.result()
            print(f"[judge] {name} done: {scores[name]:.4f}")

    average_score = (scores["deepseek"] + scores["gemini"] + scores["claude"]) / 3

    return {
        "BLEU": bleu_scores["bleu"],
        "METEOR": meteor_scores["meteor"],
        "ROUGE-L": rouge_scores["rougeL"],
        "DEEPSEEK_EVAL": scores["deepseek"],
        "GEMINI_EVAL": scores["gemini"],
        "CLAUDE_EVAL": scores["claude"],
        "AVERAGE_EVAL": average_score,
    }


def inference(messages, model, processor, max_new_tokens):
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
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # deterministic decoding
            num_beams=1,       # greedy (highest-prob token each step)
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


def eval_main(model, processor, data_path, model_size, evaluation_category, output_path, model_dir, checkpoint, max_workers=4):
    # 'category' answers are short labels (A/B/C/D); 'description' answers are full paragraphs.
    max_new_tokens = 8 if evaluation_category == "category" else 512

    references = []
    predictions = []
    video_paths = []

    with open(data_path, "r") as f:
        data = json.load(f)

    for conversation in data:
        video_path = conversation["video"]
        sys_prompt = conversation["system_prompt"]
        question = conversation["conversations"][0]["value"].split("\n")[-1]
        ref_ans = conversation["conversations"][1]["value"]

        video_paths.append(video_path)
        references.append(ref_ans)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": question},
                ],
            },
        ]

        output_text = inference(messages, model, processor, max_new_tokens)
        predictions.append(output_text[0])
        print(f"video_path: {video_path}")
        print(f"question: {messages}")
        print(f"output_text: {output_text}")

    if evaluation_category == "category":
        results = quantitative_eval(references, predictions)
        out_file_name = f"evaluation_results_qwen3_{model_size}.txt"
        results_label = "For quantitative evaluation:"
    else:
        results = text_eval(
            references,
            predictions,
            video_paths,
            output_path,
            evaluation_category,
            model_size=model_size,
            checkpoint=checkpoint,
            max_workers=max_workers,
        )
        out_file_name = f"description_evaluation_results_qwen3_{model_size}.txt"
        results_label = "For text-based evaluation:"

    with open(output_path / out_file_name, "a") as out_file:
        out_file.write(f"Evaluation Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        out_file.write(f"Model Directory: {model_dir}\n")
        out_file.write(f"\n{'='*40}\n")
        out_file.write(f"Category: {evaluation_category}\n")
        out_file.write(f"{'='*40}\n")
        out_file.write(f"{results_label}\n")
        out_file.write(json.dumps(results, indent=4))
        out_file.write("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a full-parameter fine-tuned Qwen3VL model")
    parser.add_argument("--test-data-path", type=Path, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Name or path of the base VLM model")
    parser.add_argument("--model-root", "--adapter-root", dest="model_root", type=Path, default=None, help="Path to the full fine-tuned model root (or checkpoint parent)")
    parser.add_argument("--output-path", "-o", type=Path, required=True, help="Path to save evaluation results")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint step to load the full fine-tuned model from")
    parser.add_argument(
        "--evaluation-category", "-e", type=str,
        default="category",
        choices=["description", "category"],
        help="Evaluation category: 'category' runs classification metrics, 'description' runs text-generation metrics.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of parallel LLM-judge requests (only used for 'description' evaluation).",
    )
    args = parser.parse_args()

    base_model_id = args.base_model
    test_data_path = args.test_data_path
    model_size = base_model_id.split("-")[-2]  # e.g., "8B"

    if args.model_root is None:
        model_dir = base_model_id
    elif args.checkpoint is None:
        model_dir = str(args.model_root)
    else:
        model_dir = str(args.model_root / f"checkpoint-{args.checkpoint}")
    print(f"Loading full fine-tuned model from: {model_dir}")

    args.output_path.mkdir(parents=True, exist_ok=True)

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
    except OSError:
        print(f"WARNING: Processor files missing in {model_dir}. Falling back to base model processor: {base_model_id}")
        processor = AutoProcessor.from_pretrained(base_model_id, fix_mistral_regex=True)
        print("DEBUG: Processor loaded from base model.")

    eval_main(
        model=model,
        processor=processor,
        data_path=test_data_path,
        model_size=model_size,
        evaluation_category=args.evaluation_category,
        output_path=args.output_path,
        model_dir=model_dir,
        checkpoint=args.checkpoint,
        max_workers=args.max_workers,
    )
