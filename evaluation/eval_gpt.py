import evaluate
# from transformers import AutoConfig, Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText
import torch
import argparse
from pathlib import Path
from peft import PeftModel
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from QA_pair_database import QA_database
from GPT_Eval import eval_main as g_eval
from DeepSeek_Eval import eval_main as ds_eval
from Gemini_Eval import eval_main as gem_eval
from Qwen_Eval import eval_main as qwen_eval
from Claude_Eval import eval_main as claude_eval

QAB = QA_database()
GPT_MODEL_ID = "gpt-5-mini-2025-08-07"
DEEPSEEK_MODEL_ID = "deepseek-v4-pro"
GEMINI_MODEL_ID = "gemini-3-flash-preview"
QWEN_MODEL_ID = "qwen3.5-plus"
CLAUDE_MODEL_ID = "claude-opus-4-6"


def _parse_gpt_result(raw_result):
    if isinstance(raw_result, str):
        try:
            return json.loads(raw_result)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON format: {raw_result}") from exc
    if isinstance(raw_result, dict):
        return raw_result
    raise ValueError(f"Unsupported GPT result type: {type(raw_result)}")


def _score_single_sample(ref, pred, max_retries=3, retry_base_delay=2.0):
    last_error = None
    for attempt in range(max_retries):
        try:
            raw_result = g_eval(GPT_MODEL_ID, pred, ref)
            return _parse_gpt_result(raw_result)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break
            sleep_seconds = retry_base_delay * (2 ** attempt)
            print(f"WARNING: GPT eval failed (attempt {attempt + 1}/{max_retries}): {exc}. Retrying in {sleep_seconds:.1f}s")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"GPT eval failed after {max_retries} attempts: {last_error}")

def gpt_text_eval(
    references,
    predictions,
    video_paths,
    output_path,
    category,
    model_size="8B",
    checkpoint="final",
    max_workers=10,
):
    env_score = []
    grounding_score = []
    desc_acc_score = []
    reasoning_score = []
    total_score = []
    detail_gpt_eval_parent_directory = output_path / f"gpt_evaluation_details_{model_size}"
    detail_gpt_eval_parent_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_name = checkpoint if checkpoint not in (None, "") else "final"
    detail_file_path = detail_gpt_eval_parent_directory / f"evaluation_details_{checkpoint_name}.txt"

    if len(references) != len(predictions) or len(predictions) != len(video_paths):
        raise ValueError("references, predictions, and video_paths must have the same length")

    sample_results = [None] * len(references)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_score_single_sample, references[i], predictions[i]): i
            for i in range(len(references))
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                sample_results[index] = future.result()
                print(f"GPT eval progress: {index + 1}/{len(references)} completed")
            except Exception as exc:
                sample_results[index] = {"error": str(exc)}
                print(f"WARNING: GPT eval sample {index + 1} failed: {exc}")

    with open(detail_file_path, 'w') as out_file:
        for i, results in enumerate(sample_results):
            ref = references[i]
            pred = predictions[i]
            out_file.write(f"Sample {i+1}: {video_paths[i]}\n")
            out_file.write(f"Ground Truth: {ref}\n")
            out_file.write(f"Prediction: {pred}\n")
            out_file.write(f"Evaluation Results: {results}\n")
            out_file.write("\n")

            if not isinstance(results, dict) or "error" in results:
                continue

            env_score.append(results["environment_score"])
            grounding_score.append(results["grounding_score"])
            desc_acc_score.append(results["description_accuracy_score"])
            reasoning_score.append(results["reasoning_score"])
            total_score.append(results["total_score"])

        if len(total_score) == 0:
            out_file.write(f"\n{'='*40}\n")
            out_file.write(f"Summary for category: {category}\n")
            out_file.write(f"{'='*40}\n")
            out_file.write("No valid GPT evaluation results were produced.\n")
            out_file.write("\n")
            raise RuntimeError(f"No valid GPT evaluation results. Details saved to {detail_file_path}")

        out_file.write(f"\n{'='*40}\n")
        out_file.write(f"Summary for category: {category}\n")
        out_file.write(f"{'='*40}\n")
        summary = {
            "environment_score_avg": sum(env_score) / len(env_score),
            "grounding_score_avg": sum(grounding_score) / len(grounding_score),
            "description_accuracy_score_avg": sum(desc_acc_score) / len(desc_acc_score),
            "reasoning_score_avg": sum(reasoning_score) / len(reasoning_score),
            "total_score_avg": sum(total_score) / len(total_score),
        }
        out_file.write(json.dumps(summary, indent=4))
        out_file.write("\n")
        print(f"Saved GPT per-sample details to: {detail_file_path}")

        return summary["total_score_avg"]


def _make_score_single_sample(judge_fn, model_id, label):
    """Return a callable that scores one sample with the given judge."""
    def _score(ref, pred, max_retries=5, retry_base_delay=2.0):
        last_error = None
        for attempt in range(max_retries):
            try:
                raw_result = judge_fn(model_id, pred, ref)
                return _parse_gpt_result(raw_result)
            except Exception as exc:
                last_error = exc
                if attempt == max_retries - 1:
                    break
                # Rate limit (429) or overloaded (529): wait a full minute
                exc_str = str(exc).lower()
                is_rate_limit = "429" in exc_str or "rate_limit" in exc_str or "529" in exc_str or "overloaded" in exc_str
                sleep_seconds = 60.0 if is_rate_limit else retry_base_delay * (2 ** attempt)
                print(
                    f"WARNING: {label} eval failed (attempt {attempt + 1}/{max_retries}): "
                    f"{exc}. Retrying in {sleep_seconds:.1f}s"
                )
                time.sleep(sleep_seconds)
        raise RuntimeError(f"{label} eval failed after {max_retries} attempts: {last_error}")
    return _score


def _run_judge_eval(
    score_fn,
    references,
    predictions,
    video_paths,
    output_path,
    category,
    model_size,
    checkpoint,
    label,
    detail_subdir,
    max_workers=4,
):
    """Shared evaluation loop used by deepseek/gemini/qwen_text_eval."""
    env_score = []
    grounding_score = []
    desc_acc_score = []
    reasoning_score = []
    total_score = []

    detail_dir = output_path / f"{detail_subdir}_{model_size}"
    detail_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = checkpoint if checkpoint not in (None, "") else "final"
    detail_file_path = detail_dir / f"evaluation_details_{checkpoint_name}.txt"

    if len(references) != len(predictions) or len(predictions) != len(video_paths):
        raise ValueError("references, predictions, and video_paths must have the same length")

    sample_results = [None] * len(references)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(score_fn, references[i], predictions[i]): i
            for i in range(len(references))
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                sample_results[index] = future.result()
                print(f"{label} eval progress: {index + 1}/{len(references)} completed")
            except Exception as exc:
                sample_results[index] = {"error": str(exc)}
                print(f"WARNING: {label} eval sample {index + 1} failed: {exc}")

    with open(detail_file_path, 'w') as out_file:
        for i, results in enumerate(sample_results):
            out_file.write(f"Sample {i+1}: {video_paths[i]}\n")
            out_file.write(f"Ground Truth: {references[i]}\n")
            out_file.write(f"Prediction: {predictions[i]}\n")
            out_file.write(f"Evaluation Results: {results}\n")
            out_file.write("\n")

            if not isinstance(results, dict) or "error" in results:
                continue

            env_score.append(results["environment_score"])
            grounding_score.append(results["grounding_score"])
            desc_acc_score.append(results["description_accuracy_score"])
            reasoning_score.append(results["reasoning_score"])
            total_score.append(results["total_score"])

        if len(total_score) == 0:
            out_file.write(f"\n{'='*40}\n")
            out_file.write(f"Summary for category: {category}\n")
            out_file.write(f"{'='*40}\n")
            out_file.write(f"No valid {label} evaluation results were produced.\n\n")
            raise RuntimeError(
                f"No valid {label} evaluation results. Details saved to {detail_file_path}"
            )

        out_file.write(f"\n{'='*40}\n")
        out_file.write(f"Summary for category: {category}\n")
        out_file.write(f"{'='*40}\n")
        summary = {
            "environment_score_avg": sum(env_score) / len(env_score),
            "grounding_score_avg": sum(grounding_score) / len(grounding_score),
            "description_accuracy_score_avg": sum(desc_acc_score) / len(desc_acc_score),
            "reasoning_score_avg": sum(reasoning_score) / len(reasoning_score),
            "total_score_avg": sum(total_score) / len(total_score),
        }
        out_file.write(json.dumps(summary, indent=4))
        out_file.write("\n")
        print(f"Saved {label} per-sample details to: {detail_file_path}")
        return summary["total_score_avg"]


def deepseek_text_eval(
    references,
    predictions,
    video_paths,
    output_path,
    category,
    model_size="8B",
    checkpoint="final",
    max_workers=4,
):
    score_fn = _make_score_single_sample(ds_eval, DEEPSEEK_MODEL_ID, "DeepSeek")
    return _run_judge_eval(
        score_fn, references, predictions, video_paths,
        output_path, category, model_size, checkpoint,
        label="DeepSeek", detail_subdir="deepseek_evaluation_details",
        max_workers=max_workers,
    )


def gemini_text_eval(
    references,
    predictions,
    video_paths,
    output_path,
    category,
    model_size="8B",
    checkpoint="final",
    max_workers=4,
):
    score_fn = _make_score_single_sample(gem_eval, GEMINI_MODEL_ID, "Gemini")
    return _run_judge_eval(
        score_fn, references, predictions, video_paths,
        output_path, category, model_size, checkpoint,
        label="Gemini", detail_subdir="gemini_evaluation_details",
        max_workers=max_workers,
    )


def qwen_text_eval(
    references,
    predictions,
    video_paths,
    output_path,
    category,
    model_size="8B",
    checkpoint="final",
    max_workers=4,
):
    score_fn = _make_score_single_sample(qwen_eval, QWEN_MODEL_ID, "Qwen")
    return _run_judge_eval(
        score_fn, references, predictions, video_paths,
        output_path, category, model_size, checkpoint,
        label="Qwen", detail_subdir="qwen_evaluation_details",
        max_workers=max_workers,
    )


def claude_text_eval(
    references,
    predictions,
    video_paths,
    output_path,
    category,
    model_size="8B",
    checkpoint="final",
    max_workers=2,
):
    score_fn = _make_score_single_sample(claude_eval, CLAUDE_MODEL_ID, "Claude")
    return _run_judge_eval(
        score_fn, references, predictions, video_paths,
        output_path, category, model_size, checkpoint,
        label="Claude", detail_subdir="claude_evaluation_details",
        max_workers=max_workers,
    )


def inference(messages, model, processor):
    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def eval_main(model, processor, data_path, evaluation_category, output_path):
    # Load evaluation dataset
    reference = []
    prediction = []
    video_paths = []

    with open(data_path, 'r') as f:
        data = json.load(f)
        for conversation in data:
            messages = []
            video_path = conversation['video']
            question = conversation['conversations'][0]['value'].split("\n")[-1]
            ref_ans = conversation['conversations'][1]['value']
            question_cat = QAB.question_type_query(question)

            if question_cat != evaluation_category:
                continue

            video_paths.append(video_path)
            
            if question_cat == "description":
                question = "Do you see any anomalous driving behavior in the video? If yes, please describe it."

            reference.append(ref_ans)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                        },
                        {"type": "text", "text": question},
                    ],
                })
            
            output_text = inference(messages, model, processor)
            prediction.append(output_text[0])
            print(f"question: {question}")
            print(f"output_text: {output_text}")
        
        # Evaluation
        if evaluation_category == "description":
            results = gpt_text_eval(reference, prediction, video_paths, output_path, evaluation_category)
            with open(output_path / f"evaluation_results_qwen3_8b.txt", 'a') as out_file:
                out_file.write(f"\n{'='*40}\n")
                out_file.write(f"Category: {evaluation_category}\n")
                out_file.write(f"{'='*40}\n")
                out_file.write(f"For text-based evaluation:\n")
                out_file.write(json.dumps(results, indent=4))
                out_file.write("\n\n")

if __name__ == "__main__":
    data_path = "/workspace/QWEN3-VL/datasets/CARMEL_VAD/carmel_vad_test_rewritten.json"
    base_model_id = "Qwen/Qwen3-VL-4B-Instruct"
    adapter_dir = "/output/sft_qwen3_4b_carmel_vad_second"
    parser = argparse.ArgumentParser(description="Evaluate Qwen3VL Model")
    parser.add_argument("--output-path","-o", type=Path, required=True, help="Path to save evaluation results")
    parser.add_argument("--evaluation-category","-e", type=str,
    default="description",
    choices=['description','analysis','severity','category','classification'],
    help="Evaluation category.")

    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
    )

    model.eval()

    # Load processor (handles images, videos, text prompts, etc.)
    processor = AutoProcessor.from_pretrained("/output/sft_qwen3_4b_carmel_vad_second")
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    eval_main(model=model, processor=processor, data_path=data_path, evaluation_category=args.evaluation_category, output_path=args.output_path)