# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Prompt LoRA Trainer** — QLoRA fine-tuning pipeline for training Qwen3-4B/8B to generate cinematic video diffusion prompts (ComfyUI, LTX-Video, WanVideo). Includes full pipeline from data processing through GGUF export and Ollama deployment.

## Project Status

Active development on `feat/dataset-v2-concise` branch. Three LoRA adapters trained and published:
- `Limbicnation/qwen3-4b-prompt-lora` (general video prompts, v1)
- `Limbicnation/qwen3-4b-deforum-prompt-lora` (deforum cinematic prompts, v1)
- `Limbicnation/qwen3-4b-deforum-prompt-lora-v2` (deforum v2, varied-length outputs — **overfits, needs retraining with eval split**)

**v2 Training Review (2026-02-16):** The v2 run (checkpoint-909, 51 min) achieved 99.1% train accuracy but has no eval split — inference testing confirms memorization/overfitting. Config updated with eval split + early stopping for next run.

## Development Setup

```bash
# Use conda env (NOT .venv-train which has CUBLAS issues)
conda activate prompt-lora-trainer  # Python 3.10, torch 2.6.0+cu124

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
python scripts/train_sft.py --config configs/sft_qwen3_4b_deforum_v2_final.yaml --dry-run
python scripts/train_sft.py --config configs/sft_qwen3_4b_deforum_v2_final.yaml

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
- **Package manager**: uv (pyproject.toml, uv.lock)

## Key Files

| File | Purpose |
|------|---------|
| `scripts/train_sft.py` | Main SFT training script (supports eval split + early stopping) |
| `scripts/train_unsloth.py` | Experimental Unsloth training variant |
| `scripts/validate_dataset.py` | Dataset format validation |
| `scripts/merge_and_convert_gguf.py` | LoRA merge + GGUF conversion |
| `convert_and_upload.sh` | Full pipeline: merge → GGUF → Ollama → HF |
| `configs/sft_qwen3_4b_deforum_v2_final.yaml` | Active v2 training config (with eval/early stopping) |
| `configs/sft_qwen3_4b_deforum.yaml` | Deforum v1 config (reference) |
| `Modelfile` | Ollama model definition (deforum v2) |
| `Modelfile.qwen3-prompt-lora` | Ollama model definition (v1) |
| `AGENTS.md` | Full agent context (architecture, patterns, commands) |

## Known Issues

- `extra_special_tokens` bug: transformers serializes as list, expects dict on reload → workaround in `convert_and_upload.sh`
- TRL 0.27.1: use `max_length` not `max_seq_length` in SFTConfig
- Small datasets converge fast, overfit after epoch 1 → **always use eval split + early stopping**
- llama.cpp cloned into repo for GGUF conversion (gitignored, not a submodule)
- torch 2.10+cu128 CUBLAS bug: use conda env with torch 2.6.0+cu124 instead of .venv-train
- v2 dataset uses custom `### Instruction: / ### Response:` format, not Qwen3's native chat template → adapter isn't composable with standard Qwen3 chat inference (works in Ollama via Modelfile)
