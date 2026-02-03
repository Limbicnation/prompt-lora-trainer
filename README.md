# prompt-lora-trainer

LoRA fine-tuning pipeline for training prompt-generation models on synthetic video/image prompt datasets.

## 🎯 Objective

Train a LoRA adapter on **Qwen3-4B/8B Instruct** to generate high-quality video prompts compatible with ComfyUI, LTX-Video, and WanVideo.

## 📊 Dataset

| Dataset | Rows | Status |
|---------|------|--------|
| [Video-Diffusion-Prompt-Style](https://huggingface.co/datasets/Limbicnation/Video-Diffusion-Prompt-Style) | 752 | ✅ Ready |

## 🚀 Quick Start

```bash
# 1. Create environment
conda create -n prompt-lora-trainer python=3.10 -y
conda activate prompt-lora-trainer

# 2. Install PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install training deps
pip install transformers accelerate datasets peft bitsandbytes trl huggingface-hub wandb python-dotenv pyyaml

# 4. Validate dataset
python scripts/validate_dataset.py --dataset Limbicnation/Video-Diffusion-Prompt-Style

# 5. Train (dry-run first)
python scripts/train_sft.py --config configs/sft_qwen3_4b.yaml --dry-run
python scripts/train_sft.py --config configs/sft_qwen3_4b.yaml
```

## 📁 Structure

```
├── configs/sft_qwen3_4b.yaml   # Training config
├── scripts/
│   ├── train_sft.py            # Main SFT script (QLoRA)
│   └── validate_dataset.py     # Dataset validator
├── pyproject.toml              # Dependencies
└── setup_env.sh                # Env setup script
```

## ⚙️ Training Config

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen3-4B-Instruct |
| LoRA r/α | 16/32 |
| Quantization | 4-bit NF4 (QLoRA) |
| Batch | 4 × 4 gradient accum |
| LR | 2e-4 (cosine) |

## 📈 Progress

- [x] Dataset (752 prompts)
- [x] Environment setup
- [x] Training scripts
- [ ] Dry-run validation
- [ ] Full training
- [ ] Push LoRA to Hub
