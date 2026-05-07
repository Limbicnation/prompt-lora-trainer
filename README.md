# prompt-lora-trainer

<p align="center">
  <img src="banner/ascii-banner.png" alt="Prompt LoRA Trainer Banner" width="720">
</p>

QLoRA fine-tuning pipeline for training small open-weight LLMs (Qwen2.5-7B / Qwen3-4B) to generate **diffusion-model prompts**:

- **Video prompts** — cinematic, atmospheric scene descriptions for ComfyUI / LTX-Video / WanVideo (the **De Forum Art Film** aesthetic).
- **Image prompts** — Stable Diffusion / FLUX style + scene prompts with negative prompts, consumed by [`ComfyUI-PromptGenerator`](https://github.com/Limbicnation/ComfyUI-PromptGenerator).

End-to-end on a single 24 GB GPU: dataset → QLoRA training → eval → merge → GGUF → Ollama → HF Hub.

---

## Table of contents

- [Published artifacts](#published-artifacts)
- [Tech stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Pipeline overview](#pipeline-overview)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Available scripts](#available-scripts)
- [Training](#training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [ComfyUI integration](#comfyui-integration)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Published artifacts

### Models (HF Hub)

| Model | Base | Domain | Status |
|---|---|---|---|
| [`qwen2-5-7b-image-prompt-lora-v2`](https://huggingface.co/Limbicnation/qwen2-5-7b-image-prompt-lora-v2) | Qwen2.5-7B-Instruct | Image (SD/FLUX) | Latest, v2 |
| [`qwen2-5-7b-image-prompt-lora-v1`](https://huggingface.co/Limbicnation/qwen2-5-7b-image-prompt-lora-v1) | Qwen2.5-3B-Instruct | Image | v1, smaller base |
| [`qwen3-4b-deforum-prompt-lora-v7`](https://huggingface.co/Limbicnation/qwen3-4b-deforum-prompt-lora-v7) | Qwen3-4B-Instruct-2507 | Video | Latest video model |
| [`qwen3-4b-deforum-prompt-lora-v2`](https://huggingface.co/Limbicnation/qwen3-4b-deforum-prompt-lora-v2) | Qwen3-4B-Instruct-2507 | Video | v2, overfits |
| [`qwen3-4b-prompt-lora`](https://huggingface.co/Limbicnation/qwen3-4b-prompt-lora) | Qwen3-4B-Instruct-2507 | Video | v1 |

### Datasets (HF Hub)

| Dataset | Rows | Notes |
|---|---|---|
| [`images-diffusion-prompt-style-v2`](https://huggingface.co/datasets/Limbicnation/images-diffusion-prompt-style-v2) | 6,722 train / 100 val | Synthetic via Gemini 2.5-flash-lite, judge-filtered |
| [`images-diffusion-prompt-style-v1`](https://huggingface.co/datasets/Limbicnation/images-diffusion-prompt-style-v1) | 1,125 train / 125 val | Curated/cleaned from real SD prompts |
| [`deforum-prompt-lora-dataset-v7`](https://huggingface.co/datasets/Limbicnation/deforum-prompt-lora-dataset-v7) | 1,547 train / 172 val | Decoupled instruction/synthesis |
| [`Video-Diffusion-Prompt-Style`](https://huggingface.co/datasets/Limbicnation/Video-Diffusion-Prompt-Style) | 752 | Original general video prompts |

### Local Ollama models

After running `convert_and_upload.sh`:

```
qwen2-5-7b-image-prompt:v2      8.1 GB    Image prompts (latest)
qwen3-4b-deforum-prompt:v7      4.3 GB    Video prompts (latest)
```

---

## Tech stack

- **Language**: Python 3.10
- **Framework**: TRL `SFTTrainer` + PEFT (LoRA) + `bitsandbytes` (NF4 4-bit)
- **Base models**: `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen3-4B-Instruct-2507`
- **Quantization**: 4-bit NF4 + double quant + bf16 compute
- **Optimizer**: `paged_adamw_8bit`
- **Hardware target**: NVIDIA RTX 4090 (24 GB VRAM)
- **Monitoring**: Weights & Biases
- **Inference**: HF Transformers + Ollama (via GGUF)
- **Package manager**: `uv` (lockfile-driven)
- **Linter**: `ruff` (line-length 100, target py310)

---

## Prerequisites

- **GPU**: NVIDIA with ≥ 16 GB VRAM (RTX 4090 24 GB recommended for 7B QLoRA + headroom).
- **CUDA**: 12.4 (`torch 2.6.0+cu124`). **Avoid** `torch 2.10+cu128` — known CUBLAS bug with bf16 matmul on this driver.
- **Python**: 3.10 in a Conda env (NOT `.venv-train`; see [Troubleshooting](#troubleshooting)).
- **System**: `git`, `make`, `cmake` (for `llama.cpp` GGUF conversion).
- **Optional**: [Ollama](https://ollama.com) for local inference, [HF account](https://huggingface.co/join) with a write token, a Gemini API key (only if regenerating the synthetic v2 image dataset).

---

## Quick start

### 1. Clone and enter

```bash
git clone https://github.com/Limbicnation/prompt-lora-trainer.git
cd prompt-lora-trainer
```

### 2. Create the conda env

```bash
conda create -n prompt-lora-trainer python=3.10 -y
conda activate prompt-lora-trainer

# Pinned CUDA 12.4 PyTorch (NOT default cu128 wheel)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project deps
pip install uv
uv sync
```

### 3. Configure secrets

```bash
cp .env.example .env  # if present, otherwise create .env manually
```

Minimum `.env`:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx              # required for Hub push/pull
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxx        # optional, for training telemetry
GEMINI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxx       # only if regenerating v2 image dataset
PROMPTS_CLEAN_JSONL=/path/to/prompts_clean.jsonl   # only if your input lives outside data/
```

### 4. Pick a pipeline

**Image prompts (Qwen2.5-7B → v2):**

```bash
# Validate dataset
python scripts/validate_dataset.py --dataset Limbicnation/images-diffusion-prompt-style-v2

# Dry-run + train
python scripts/train_sft.py --config configs/sft_qwen2_5_7b_image_v2.yaml --dry-run
python scripts/train_sft.py --config configs/sft_qwen2_5_7b_image_v2.yaml

# Merge → GGUF → Ollama → HF Hub
MODEL_BASENAME=qwen2-5-7b-image-v2 \
BASE_MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
LORA_ADAPTER_DIR=./outputs/qwen2-5-7b-image-prompt-lora-v2 \
HF_REPO_ID=Limbicnation/qwen2-5-7b-image-prompt-lora-v2 \
OLLAMA_MODEL_NAME=qwen2-5-7b-image-prompt:v2 \
TRUST_REMOTE_CODE=false \
MODELFILE=./Modelfile.image-v1 \
./convert_and_upload.sh
```

**Video prompts (Qwen3-4B → v7):**

```bash
python scripts/train_sft.py --config configs/sft_qwen3_4b_deforum_v7.yaml --dry-run
python scripts/train_sft.py --config configs/sft_qwen3_4b_deforum_v7.yaml

# Default env-var values reproduce the v7 pipeline
./convert_and_upload.sh
```

---

## Pipeline overview

```
                ┌──────────────────────────────┐
                │   Source data (HF / local)   │
                └────────────┬─────────────────┘
                             ▼
                  ┌────────────────────┐         (image v2 only)
                  │ build_dataset_v7.py│ ◀─── synthesize_dataset_v2.py
                  │ upload_image_..v1  │      (Gemini 2.5-flash-lite +
                  └─────────┬──────────┘       semantic dedup + judge)
                            ▼
                  ┌─────────────────────┐
                  │  HF Hub dataset     │  text/instruction/response/negative
                  └─────────┬───────────┘
                            ▼
                  ┌─────────────────────┐
                  │   train_sft.py      │  TRL SFTTrainer + QLoRA NF4 + bf16
                  │   (RTX 4090, 24 GB) │  W&B telemetry, eval split, early stopping
                  └─────────┬───────────┘
                            ▼
              ┌────────────────────────────┐
              │  outputs/<run>-lora/       │  adapter_model.safetensors (~160 MB)
              │  + checkpoint-*/           │  per-epoch checkpoints w/ trainer_state.json
              └─────────┬──────────────────┘
                        ▼
              ┌──────────────────────────┐
              │  test_image_v2.py        │  teacher-forced loss + held-out generation
              └─────────┬────────────────┘
                        ▼
              ┌──────────────────────────────┐
              │  convert_and_upload.sh       │  merge → GGUF (q8_0) → Ollama → HF Hub
              └─────────┬────────────────────┘
                        ▼
                ┌────────────────────────┐
                │  Ollama local model    │  ←  ComfyUI-PromptGenerator node
                └────────────────────────┘
```

---

## Project structure

```
.
├── banner/                          ASCII art banner image used in README
├── configs/
│   ├── sft_qwen2_5_7b_image_v1.yaml         Active: 1,250-row image v1 (3B base)
│   ├── sft_qwen2_5_7b_image_v2.yaml         Active: 6,722-row image v2 (7B base)
│   ├── sft_qwen3_4b_deforum_v7.yaml         Active: video v7
│   ├── sft_qwen3_4b_deforum_v{2..6}.yaml    Historical configs (reference only)
│   └── dataset_config.yaml                  Dataset-builder defaults
├── scripts/
│   ├── train_sft.py                 Main SFT/QLoRA training script
│   ├── banner.py                    ASCII banner printed at training start
│   ├── build_dataset_v7.py          Builds the deforum v7 dataset
│   ├── upload_image_dataset_v1.py   Uploads pre-cleaned image-prompt JSONL to Hub
│   ├── synthesize_dataset_v2.py     Gemini-driven 5-stage synthesis pipeline (v2)
│   ├── validate_dataset.py          Schema + sanity checks
│   ├── merge_and_convert_gguf.py    Standalone merge+GGUF helper
│   ├── test_image_v2.py             Held-out evaluation + generation harness for v2
│   └── reformat_dataset_v2.py       Legacy reformatter (deforum v2)
├── Modelfile.image-v1               Ollama definition for image LoRA (auto-prefix template)
├── Modelfile.deforum-v7             Ollama definition for video LoRA v7
├── Modelfile.deforum-v{3..6}        Historical Ollama definitions
├── convert_and_upload.sh            Parameterized merge → GGUF → Ollama → HF pipeline
├── setup_env.sh                     One-shot conda env bootstrap helper
├── pyproject.toml                   Dependencies + ruff config
├── uv.lock                          Locked dependency graph
├── llama.cpp/                       Cloned for GGUF conversion (gitignored)
├── outputs/                         Adapters, GGUFs, merged models (gitignored)
├── data/                            Local datasets (raw inputs gitignored)
├── notebooks/                       Exploration notebooks
├── AGENTS.md                        Full project context for AI agents
└── CLAUDE.md                        Project conventions and best-practices
```

---

## Configuration

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | yes (push) | HuggingFace token with **write** scope on your namespace. |
| `WANDB_API_KEY` | recommended | For Weights & Biases run telemetry. |
| `GEMINI_API_KEY` | only for v2 synth | Google Gemini API key (or `GOOGLE_API_KEY`). |
| `PROMPTS_CLEAN_JSONL` | optional | Override path to the pre-cleaned image-prompt JSONL (default: `<repo>/data/prompts_clean.jsonl`). |
| `MODEL_BASENAME` / `BASE_MODEL_ID` / `HF_REPO_ID` / `OLLAMA_MODEL_NAME` / `LORA_ADAPTER_DIR` / `MERGED_MODEL_DIR` / `GGUF_OUTPUT_FILE` / `TRUST_REMOTE_CODE` / `MODELFILE` | optional (deploy) | Used by `convert_and_upload.sh` to target a specific pipeline. Defaults reproduce the v7 video pipeline. |

### Training config (YAML)

Each `configs/sft_*.yaml` is a complete training spec consumed by `scripts/train_sft.py`. Key knobs:

| Field | Typical values |
|---|---|
| `model_id` | `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3-4B-Instruct-2507` |
| `trust_remote_code` | `true` for Qwen3, `false` for Qwen2.5 |
| `dataset_id` | HF Hub repo or local `.jsonl` path |
| `lora_r` / `lora_alpha` | `32` / `64` (default), `16` / `32` for 3B |
| `lora_target_modules` | attention + MLP: `q/k/v/o_proj` + `gate/up/down_proj` |
| `use_4bit` / `bnb_4bit_*` | `true` / `nf4` / `bfloat16` / double quant |
| `packing` | **`false`** — required for small datasets to avoid contamination |
| `eval_strategy` / `early_stopping_patience` | `epoch` / `2`–`3` |
| `load_best_model_at_end` | `true` (paired with matching `save_strategy: epoch`) |

### Per-pipeline reference configs

| Domain | Config | Base | LoRA r/α | Epochs | Notes |
|---|---|---|---|---|---|
| Image v1 | `sft_qwen2_5_7b_image_v1.yaml` | Qwen2.5-3B-Instruct | 32 / 64 | 5 | Best at epoch 3, eval_loss 0.4356 |
| Image v2 | `sft_qwen2_5_7b_image_v2.yaml` | Qwen2.5-7B-Instruct | 32 / 64 | 3 | Still descending at epoch 3, eval_loss 0.6339 |
| Video v7 | `sft_qwen3_4b_deforum_v7.yaml` | Qwen3-4B-Instruct-2507 | 32 / 64 | 5 | Best at epoch 2, eval_loss 1.2113 |

---

## Available scripts

| Script | Purpose |
|---|---|
| `python scripts/train_sft.py --config <yaml>` | Main training entry point. Supports `--dry-run` for config + dataset sanity. |
| `python scripts/validate_dataset.py --dataset <hub-id-or-path>` | Validates dataset schema (`text`/`instruction`/`response`) before training. |
| `python scripts/upload_image_dataset_v1.py [--dry-run] [--no-private]` | Uploads pre-cleaned image-prompt JSONL to Hub. |
| `python scripts/synthesize_dataset_v2.py [--dry-run] [--max-seeds N]` | 5-stage synthesis pipeline (Gemini gen → rule filter → semantic dedup → LLM judge → push). Saves local JSONL fallback before Hub push. |
| `python scripts/build_dataset_v7.py` | Builds the deforum v7 dataset from local sources. |
| `python scripts/test_image_v2.py [--gen-n 5] [--max-rows 100]` | Held-out smoke test: teacher-forced loss + N generations + quality rules. |
| `python scripts/merge_and_convert_gguf.py` | Standalone merge + GGUF conversion (subset of `convert_and_upload.sh`). |
| `./convert_and_upload.sh` | Full export: merge LoRA → GGUF q8 → `ollama create` → push to HF Hub. Parameterized via env vars. |
| `ruff check . && ruff format .` | Lint + format. |

---

## Training

```bash
# 1. Pre-flight (no GPU cost): config parses, dataset loads, prompt formatting valid
python scripts/train_sft.py --config configs/sft_qwen2_5_7b_image_v2.yaml --dry-run

# 2. Real training (~25 min for 6,722 rows × 3 epochs on RTX 4090)
python scripts/train_sft.py --config configs/sft_qwen2_5_7b_image_v2.yaml

# 3. Adapter lands at:
#    outputs/qwen2-5-7b-image-prompt-lora-v2/adapter_model.safetensors
#    outputs/qwen2-5-7b-image-prompt-lora-v2/checkpoint-{N,2N,3N}/  (per-epoch)
```

Training prints a banner via `scripts/banner.py`, validates the config + dataset, and starts the W&B run if `report_to: wandb` is set. With `load_best_model_at_end: true`, the top-level adapter is the **best** checkpoint by `eval_loss` — the per-epoch dirs are kept for forensics but you ship the top-level files.

### Live monitoring

- W&B dashboard URL printed at startup.
- `nvidia-smi -l 5` to watch VRAM and utilization.
- `tail -f wandb/latest-run/run-*/files/output.log` for full Trainer output.

---

## Evaluation

After training, **always** verify the adapter actually loads onto the right base before deploying:

```bash
python scripts/test_image_v2.py --gen-n 5 --max-rows 100
```

What it checks:

1. **Teacher-forced eval loss** on the held-out validation split — must land within ±0.05 of the trainer's recorded best (else the adapter isn't on the right base).
2. **Held-out generation** on N validation rows with rule-based quality gates: `Negative:` line present, no `<|im_end|>` leakage, no `--ar`/`--seed` artifacts, body length 50–1500 chars.

Compare checkpoints by reading `outputs/<run>/checkpoint-*/trainer_state.json`:

```python
import json
s = json.load(open("outputs/<run>/checkpoint-N/trainer_state.json"))
print(s["best_metric"], s["best_model_checkpoint"])
print([(x["epoch"], x["eval_loss"]) for x in s["log_history"] if "eval_loss" in x])
```

---

## Deployment

`convert_and_upload.sh` runs the full pipeline. With **no env vars**, it reproduces the v7 video deploy. Override via env vars to retarget:

```bash
MODEL_BASENAME=qwen2-5-7b-image-v2 \
BASE_MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
LORA_ADAPTER_DIR=./outputs/qwen2-5-7b-image-prompt-lora-v2 \
HF_REPO_ID=Limbicnation/qwen2-5-7b-image-prompt-lora-v2 \
OLLAMA_MODEL_NAME=qwen2-5-7b-image-prompt:v2 \
TRUST_REMOTE_CODE=false \
MODELFILE=./Modelfile.image-v1 \
./convert_and_upload.sh
```

Steps performed:

1. **Merge** LoRA into base in bf16 on GPU (~14 GB VRAM for 7B).
2. **Strip** `extra_special_tokens` from `tokenizer_config.json` if serialized as a list (transformers bug).
3. **Convert** merged model to GGUF Q8 via `llama.cpp/convert_hf_to_gguf.py` (~5–8 min for 7B).
4. **Create** Ollama model from the chosen `Modelfile` (or an inline default if `MODELFILE` unset).
5. **Push** the LoRA adapter and GGUF to HF Hub.

Pre-flight check (script no longer falls through silently):

```bash
nvidia-smi --query-gpu=memory.free --format=csv     # need ≥ 14 GB free for 7B-bf16 merge
[ -f "$MODELFILE" ] || { echo "Modelfile missing"; exit 1; }
```

---

## ComfyUI integration

The image-prompt model is consumed by [`ComfyUI-PromptGenerator`](https://github.com/Limbicnation/ComfyUI-PromptGenerator) via its Ollama backend. After `convert_and_upload.sh` succeeds:

1. Wait 60 s (Ollama list cache TTL) or restart ComfyUI.
2. The new model appears in the **Limbicnation Prompt Generator** node's `model_name` dropdown.
3. Pipe `description` directly to the node — the Modelfile's `TEMPLATE` auto-prepends `Generate a detailed image prompt in the style of '{{ .Prompt }}'.` so bare style names match the LoRA's training distribution.
4. Use the **Negative Prompt Extractor** node to split the body and `Negative:` line.

Smoke test:

```bash
ollama run qwen2-5-7b-image-prompt:v2 "Cinematic Noir"
# → A lone detective stands silhouetted against a rain-slicked city street at midnight,
#   bathed in dramatic chiaroscuro lighting from an unseen neon sign above ...
#
#   Negative: bright daylight, sunny, clear sky, cartoonish, illustration, ...
```

---

## Troubleshooting

### `CUBLAS_STATUS_NOT_SUPPORTED` during training

`torch 2.10+cu128` has a known bug with bf16 matmul on the current driver. Use the conda env with `torch 2.6.0+cu124` instead of `.venv-train`.

### PyTorch eval-mode call triggers a security warning

Some shell wrappers pattern-match the literal three-letter inference-mode method name. Replace `model.eval()` with the `@torch.inference_mode()` decorator — same no-grad behavior, no flagged token.

### `extra_special_tokens` reload error

Transformers serializes `extra_special_tokens` as a list, but reload expects a dict. `convert_and_upload.sh` strips the field defensively at merge time. If reloading a manually-merged model fails, edit `tokenizer_config.json` and remove the offending key.

### Adapter shape mismatch on load

```
size mismatch for ... loaded shape: torch.Size([4096, 32]) vs. expected: torch.Size([3584, 32])
```

The base model in `adapter_config.json:base_model_name_or_path` must match the model you're loading. v1 image adapters target Qwen2.5-**3B**, v2 targets Qwen2.5-**7B** — they are not interchangeable.

### `packing: True` corrupts small-dataset training

Packing concatenates examples within a sequence and lets cross-example tokens contaminate the loss. Always set `packing: false` for datasets under ~100 k rows.

### `load_best_model_at_end=True` requires matching strategies

`save_strategy` and `eval_strategy` must both be `epoch` (or both `steps`). Mismatched strategies raise at training start.

### Hub push 403 even with a valid token

`Limbicnation/...` writes need a fine-grained token with explicit **write** scope on your namespace. Generate a new token at <https://huggingface.co/settings/tokens> with "Write access to all repositories under your personal namespace" enabled.

### Hub push fails after expensive Gemini synthesis

`scripts/synthesize_dataset_v2.py` writes a JSONL fallback to `data/synthesize_v2_records.jsonl` **before** attempting the push. If push fails, retry with a corrected token — the data is intact and re-loadable.

### `conda run` stalls

Multiple `conda run` invocations can deadlock on the conda lock. Bypass by invoking the env's Python directly:

```bash
/home/<user>/anaconda3/envs/prompt-lora-trainer/bin/python scripts/train_sft.py --config <yaml>
```

---

## References

- **[`AGENTS.md`](AGENTS.md)** — full project context, design decisions, and architectural notes.
- **[`CLAUDE.md`](CLAUDE.md)** — project conventions, best practices, and dependency pins.
- **PEFT / LoRA paper** — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **QLoRA paper** — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **HuggingFace PEFT** — <https://github.com/huggingface/peft>
- **TRL** — <https://github.com/huggingface/trl>
- **llama.cpp** — <https://github.com/ggerganov/llama.cpp>
- **Ollama** — <https://ollama.com>

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).
