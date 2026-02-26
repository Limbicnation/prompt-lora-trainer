# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Prompt LoRA Trainer** — QLoRA fine-tuning pipeline for training Qwen3-4B/8B to generate cinematic video diffusion prompts (ComfyUI, LTX-Video, WanVideo). Includes full pipeline from data processing through GGUF export and Ollama deployment.

## Project Status

Active development on `main` branch. Published LoRA adapters:

- `Limbicnation/qwen3-4b-deforum-prompt-lora-v7` — **Latest** (1,547 rows, eval split + early stopping)
- `Limbicnation/qwen3-4b-prompt-lora` (general video prompts, v1)
- `Limbicnation/qwen3-4b-deforum-prompt-lora-v2` (deforum v2 — overfits, superseded by v7)

**Verified Dependency Versions (2026-02-26):** torch 2.6.0+cu124, transformers 4.57.6, peft 0.18.1, trl 0.27.1, bitsandbytes 0.49.1, datasets 4.5.0.

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
python scripts/validate_dataset.py --dataset Limbicnation/deforum-prompt-lora-dataset-v7

# Train (dry-run first)
python scripts/train_sft.py --config configs/sft_qwen3_4b_deforum_v7.yaml --dry-run
python scripts/train_sft.py --config configs/sft_qwen3_4b_deforum_v7.yaml

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
| `scripts/train_sft.py` | Main SFT training script (displays CLI banner, supports eval split + early stopping) |
| `scripts/banner.py` | ASCII art CLI banner (auto-loaded by train_sft.py, cosmetic — failures never block training) |
| `scripts/validate_dataset.py` | Dataset format validation |
| `scripts/merge_and_convert_gguf.py` | LoRA merge + GGUF conversion |
| `scripts/build_dataset_v7.py` | Dataset builder (current) |
| `convert_and_upload.sh` | Full pipeline: merge → GGUF → Ollama → HF |
| `configs/sft_qwen3_4b_deforum_v7.yaml` | Active v7 training config (packing=false, expanded LoRA targets, eval + early stopping) |
| `banner/ascii-banner.png` | Banner image for README branding |
| `Modelfile.deforum-v7` | Ollama model definition (v7, auto-prefix template) |
| `AGENTS.md` | Full agent context (architecture, patterns, commands) |

## Known Issues

- `extra_special_tokens` bug: transformers serializes as list, expects dict on reload → workaround in `convert_and_upload.sh`
- TRL 0.27.1: use `max_length` not `max_seq_length` in SFTConfig
- Small datasets converge fast, overfit after epoch 1 → **always use eval split + early stopping**
- llama.cpp cloned into repo for GGUF conversion (gitignored, not a submodule)
- torch 2.10+cu128 CUBLAS bug: use conda env with torch 2.6.0+cu124 instead of .venv-train
- v2 dataset uses custom `### Instruction: / ### Response:` format, not Qwen3's native chat template → adapter isn't composable with standard Qwen3 chat inference (works in Ollama via Modelfile)

## Best Practices

- **Always dry-run before training**: `--dry-run` validates config, dataset loading, and prompt formatting without GPU cost
- **Banner pattern**: `scripts/banner.py` is loaded via `from banner import print_banner` inside a `try/except Exception` — cosmetic imports should never block core functionality
- **Use conda env, not venvs**: The `.venv-train` has CUBLAS issues with torch 2.10+cu128; conda with torch 2.6.0+cu124 is stable
- **Feature branches**: Use `feat/*` branches, push with `--force-with-lease` after amending commits
- **conda run can stall**: If multiple `conda run` processes hold locks, use the env Python directly: `/home/gero/anaconda3/envs/prompt-lora-trainer/bin/python`
