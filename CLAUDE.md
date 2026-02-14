# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Prompt LoRA Trainer** — QLoRA fine-tuning pipeline for training Qwen3-4B/8B to generate cinematic video diffusion prompts (ComfyUI, LTX-Video, WanVideo). Includes full pipeline from data processing through GGUF export and Ollama deployment.

## Project Status

Active development. Two LoRA adapters trained and published to HuggingFace Hub:
- `Limbicnation/qwen3-4b-prompt-lora` (general video prompts)
- `Limbicnation/qwen3-4b-deforum-prompt-lora` (deforum cinematic prompts)

Current branch: `feat/gguf-export-pipeline` — adds merge, GGUF conversion, and Ollama upload pipeline.

## Development Setup

```bash
# Environment uses uv with .venv-train (Python 3.13)
source .venv-train/bin/activate

# Install dependencies
pip install -e .

# Required environment variables (.env file)
HF_TOKEN=hf_...
WANDB_API_KEY=...
```

## Key Commands

```bash
# Validate dataset
python scripts/validate_dataset.py --dataset Limbicnation/Video-Diffusion-Prompt-Style

# Train (dry-run first)
python scripts/train_sft.py --config configs/sft_qwen3_4b_deforum.yaml --dry-run
python scripts/train_sft.py --config configs/sft_qwen3_4b_deforum.yaml

# Full export pipeline (merge → GGUF → Ollama → HF upload)
./convert_and_upload.sh

# Merge LoRA only
python scripts/merge_and_convert_gguf.py --output-dir ./outputs/merged

# Lint
ruff check . && ruff format .
```

## Architecture

- **Framework**: TRL SFTTrainer + PEFT + bitsandbytes (QLoRA, NF4, bf16)
- **Base model**: `Qwen/Qwen3-4B-Instruct-2507`
- **Hardware target**: RTX 4090 (24GB VRAM)
- **Monitoring**: Weights & Biases
- **Configs**: YAML files in `configs/`
- **Linter**: ruff (line-length 100, target py310)

## Key Files

| File | Purpose |
|------|---------|
| `scripts/train_sft.py` | Main SFT training script |
| `scripts/validate_dataset.py` | Dataset format validation |
| `scripts/merge_and_convert_gguf.py` | LoRA merge + GGUF conversion |
| `convert_and_upload.sh` | Full pipeline: merge → GGUF → Ollama → HF |
| `configs/sft_qwen3_4b_deforum.yaml` | Active training config |
| `Modelfile.qwen3-prompt-lora` | Ollama model definition |
| `AGENTS.md` | Full agent context (architecture, patterns, commands) |
| `IMPLEMENTATION_PLAN.md` | Original training plan |

## Known Issues

- `extra_special_tokens` bug: transformers serializes as list, expects dict on reload → workaround in `convert_and_upload.sh`
- TRL 0.27.1: use `max_length` not `max_seq_length` in SFTConfig
- Small datasets (752 rows) converge fast, overfit after epoch 1 → use early stopping
- llama.cpp cloned into repo for GGUF conversion (gitignored, not a submodule)
