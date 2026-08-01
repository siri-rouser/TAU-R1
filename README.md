# [TAU-R1: Visual Language Model for Traffic Anomaly Understanding](https://arxiv.org/abs/2603.19098)

**TAU-R1 (Traffic Anomaly Understanding - R1)** is a VLM-based two-layer hierarchical framework for understanding traffic anomalies in real-world roundabout scenarios.

## Environment Setup

Run the commands below from the repository root. You need an NVIDIA GPU with a compatible driver for training.

### Option 1: Docker image (recommended)

The image uses `vllm/vllm-openai:v0.10.2`, installs the runtime dependencies, and installs the vendored [transformers](transformers), [trl](trl), and [qwen-vl-utils](qwen-vl-utils) packages in editable mode.

Build the image with Docker:

```bash
docker build -f docker/Dockerfile.taur1 -t tau-r1:latest .
```

### Option 2: Local Python environment

**Prerequisites:** [Conda](https://docs.conda.io/en/latest/miniconda.html) or Miniconda/Mamba, Python 3.12, an NVIDIA GPU, and CUDA 12.8-compatible drivers.

Create and activate an environment:

```bash
conda create -n tau-r1 python=3.12 -y
conda activate tau-r1
python -m pip install --upgrade pip
```

Install PyTorch, vLLM, and the runtime dependencies:

```bash
python -m pip install \
  "torch==2.8.0" \
  "torchvision==0.23.0" \
  "torchaudio==2.8.0" \
  "vllm==0.10.2" \
  --extra-index-url https://download.pytorch.org/whl/cu128

python -m pip install --no-build-isolation "flash-attn==2.7.4.post1"

python -m pip install \
  "torchcodec==0.7.0" \
  "gradio==5.49.1" \
  "transformers-stream-generator==0.0.5" \
  "accelerate==1.7.0" \
  "datasets==4.5.0" \
  "deepspeed==0.17.5" \
  "evaluate==0.4.6" \
  "absl-py" \
  "nltk" \
  "rouge-score" \
  "sacrebleu" \
  "anthropic" \
  "openai==2.1.0" \
  "peft==0.17.1" \
  "wandb==0.24.1" \
  "huggingface-hub==0.35.3"
```

Install the vendored packages in editable mode:

```bash
python -m pip install -e ./transformers -e ./qwen-vl-utils -e ./trl
```

Download the NLTK data used by METEOR:

```bash
python -m nltk.downloader wordnet
```

## Dataset and Models

- **Dataset:** [Roundabout-TAU](https://huggingface.co/datasets/yl4300/Roundabout-TAU)
- **Summariser:** [TAU-R1-Summarizer](https://huggingface.co/yl4300/TAU-R1-Summarizer)
- **Classifier:** [TAU-R1-Classifier](https://huggingface.co/yl4300/TAU-R1-Classifier)


## Training

Training follows a two-stage SFT pipeline followed by GRPO post-training, applied separately to the Classifier (Qwen3-VL-2B) and Summariser (Qwen3-VL-8B) models. Training scripts and configs are provided under `train/scripts/`.

## Evaluation

The two full-parameter evaluation workflows are combined in `evaluation/eval_full_parameter.py`. Select the workflow with `--evaluation-category`.

For classifier:

```bash
python evaluation/eval_full_parameter.py \
  --test-data-path "/workspace/TAU-R1/datasets/Roundabout_TAU/roundabout_tau_category_qa_pairs_test.json" \
  --base-model "Qwen/Qwen3-VL-2B-Instruct" \
  --model-root "/output/sft_qwen3_2b_full_parameter_grpo" \
  --output-path "/workspace/TAU-R1/evaluation/" \
  --evaluation-category "category"
```

For summariser:

```bash
python evaluation/eval_full_parameter.py \
  --test-data-path "/workspace/TAU-R1/datasets/Roundabout_TAU/roundabout_tau_summary_qa_pairs_test.json" \
  --base-model "Qwen/Qwen3-VL-8B-Instruct" \
  --model-root "/output/sft_qwen3_8b_stage2_grpo_full_parameter" \
  --output-path "/workspace/TAU-R1/evaluation/" \
  --evaluation-category "description"
```

Description evaluation also calls the DeepSeek, Gemini, and Claude API to calculate the LLM Score. Set their API keys in the environment before running it:

```bash
export DEEPSEEK_API_KEY="..."
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

## Single-video inference

```bash
python evaluation/eval_onesample.py \
  --test-video-path "/data/your_video.mp4" \
  --base-model "Qwen/Qwen3-VL-8B-Instruct" \
  --model-root "/output/sft_qwen3_8b_stage2_grpo_full_parameter" \
  --sys-prompt "You are an expert driving behavior analyst. Ground your answer only in visible evidence." \
  --question "Your question"
```

## Citation

If you find this project helpful, please cite:

```bibtex
@misc{lin2026taur1visuallanguagemodel,
  title        = {TAU-R1: Visual Language Model for Traffic Anomaly Understanding},
  author       = {Yuqiang Lin and Kehua Chen and Sam Lockyer and Arjun Yadav and Mingxuan Sui and Shucheng Zhang and Yan Shi and Bingzhang Wang and Yuang Zhang and Markus Zarbock and Florain Stanek and Adrian Evans and Wenbin Li and Yinhai Wang and Nic Zhang},
  year         = {2026},
  eprint       = {2603.19098},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url          = {https://arxiv.org/abs/2603.19098}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
