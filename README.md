# prompt-lora-trainer

QLoRA fine-tuning pipeline for training Qwen3-4B to generate cinematic video diffusion prompts. Specializes in the **De Forum Art Film** aesthetic — noir-influenced, atmospheric, psychologically minimal.

## Published Models

| Model | Dataset | Status |
|-------|---------|--------|
| [qwen3-4b-deforum-prompt-lora-v4](https://huggingface.co/Limbicnation/qwen3-4b-deforum-prompt-lora-v4) | deforum-v4 (Ollama-synthesized) | ✅ Latest |
| [qwen3-4b-deforum-prompt-lora-v3](https://huggingface.co/Limbicnation/qwen3-4b-deforum-prompt-lora-v3) | deforum-v3.1 | ✅ Stable |
| [qwen3-4b-prompt-lora](https://huggingface.co/Limbicnation/qwen3-4b-prompt-lora) | Video-Diffusion-Prompt-Style | ✅ v1 |

## Datasets

| Dataset | Rows | Notes |
|---------|------|-------|
| [deforum-prompt-lora-dataset-v4](https://huggingface.co/datasets/Limbicnation/deforum-prompt-lora-dataset-v4) | ~5,500 | Synthesized via qwen3-deforum-v3 Ollama model |
| [deforum-prompt-lora-dataset-v3.1](https://huggingface.co/datasets/Limbicnation/deforum-prompt-lora-dataset-v3.1) | ~3,000 | Creative Writing + Gutenberg extraction |
| [Video-Diffusion-Prompt-Style](https://huggingface.co/datasets/Limbicnation/Video-Diffusion-Prompt-Style) | 752 | Original general video prompts |

## Quick Start

```bash
# 1. Create environment (conda required — .venv-train has CUBLAS issues on cu128)
conda create -n prompt-lora-trainer python=3.10 -y
conda activate prompt-lora-trainer

# 2. Install PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install training deps via uv
pip install uv
uv sync

# 4. Validate dataset
uv run scripts/validate_dataset.py --dataset Limbicnation/deforum-prompt-lora-dataset-v4

# 5. Train (dry-run first)
uv run scripts/train_sft.py --config configs/sft_qwen3_4b_deforum_v4.yaml --dry-run
uv run scripts/train_sft.py --config configs/sft_qwen3_4b_deforum_v4.yaml

# 6. Export: merge → GGUF → Ollama → HF Hub
./convert_and_upload.sh
```

## 📁 Structure

```
├── configs/
│   ├── sft_qwen3_4b_deforum_v4.yaml    # Active: v4 training config
│   ├── sft_qwen3_4b_deforum_v3.yaml    # v3 config (reference)
│   └── sft_qwen3_4b.yaml               # v1 config (reference)
├── scripts/
│   ├── train_sft.py                    # Main SFT script (QLoRA + early stopping)
│   ├── build_dataset_v4.py             # Dataset builder (Ollama synthesis)
│   ├── build_dataset_v3_extraction.py  # Dataset builder (text extraction only)
│   ├── merge_and_convert_gguf.py       # LoRA merge + GGUF conversion
│   └── validate_dataset.py             # Dataset format validator
├── Modelfile.deforum-v4                # Ollama model definition (v4)
├── Modelfile.deforum-v3                # Ollama model definition (v3)
├── convert_and_upload.sh               # Full export pipeline
└── pyproject.toml                      # Dependencies (uv)
```

## ⚙️ Training Config (v4)

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen3-4B-Instruct-2507 |
| LoRA r/α | 16/32 |
| Target modules | q/k/v/o_proj (attention only) |
| Quantization | 4-bit NF4 (QLoRA, bf16 compute) |
| Batch | 2 × 4 gradient accum |
| LR | 2e-4 (cosine) |
| Epochs | 3 (early stopping, patience=2) |
| Eval | Every 10 steps |

## Ollama Deployment

After GGUF export:

```bash
# Create Ollama model
ollama create qwen3-deforum-v4 -f Modelfile.deforum-v4

# Run
ollama run qwen3-deforum-v4 "Generate a cinematic prompt for a rain-soaked alley at dusk"
```

## Known Issues

- `extra_special_tokens` serialization bug: `convert_and_upload.sh` deletes the field at merge time
- Use conda env, not `.venv-train` — the latter has torch 2.10+cu128 which causes CUBLAS errors on this driver
- Small datasets overfit fast; always use eval split + early stopping (already in all v3+ configs)
