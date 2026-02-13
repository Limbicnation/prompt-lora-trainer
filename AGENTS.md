# AGENTS.md

> **Agent Context File**: This document provides essential context for AI coding agents working on the `prompt-lora-trainer` project. It describes the architecture, conventions, and workflows used in this codebase.

---

## Project Overview

`prompt-lora-trainer` is a LoRA (Low-Rank Adaptation) fine-tuning pipeline for training prompt-generation models on synthetic video/image prompt datasets. The primary goal is to train LoRA adapters on **Qwen3-4B/8B Instruct** to generate high-quality video prompts compatible with ComfyUI, LTX-Video, and WanVideo.

### Key Objectives

- Fine-tune Qwen3 models for video diffusion prompt generation
- Support QLoRA (4-bit quantization) for training on consumer GPUs (RTX 4090 / 24GB VRAM)
- Export trained models to HuggingFace Hub and GGUF format for Ollama deployment
- Generate cinematic prompts with proper technical parameters (aspect ratio, guidance scale, seeds)

### Target Datasets

| Dataset | Rows | Status | Purpose |
|---------|------|--------|---------|
| `Limbicnation/Video-Diffusion-Prompt-Style` | 752 | ✅ Ready | General video prompts |
| `Limbicnation/deforum-prompt-lora-dataset` | Variable | 🔄 Custom | De Forum cinematic prompts |

---

## Technology Stack

### Core Dependencies

| Category | Libraries | Version |
|----------|-----------|---------|
| Deep Learning | `torch`, `torchvision`, `torchaudio` | >=2.2.0 |
| Transformers | `transformers`, `accelerate` | >=4.40.0, >=0.28.0 |
| LoRA/PEFT | `peft`, `bitsandbytes` | >=0.10.0, >=0.43.0 |
| Training Framework | `trl` | >=0.8.0 |
| Data | `datasets` | >=2.18.0 |
| Hub & Auth | `huggingface-hub` | >=0.22.0 |
| Monitoring | `wandb`, `trackio` | >=0.16.0 |
| Utilities | `python-dotenv`, `pyyaml`, `tqdm` | >=1.0.0 |

### Development Tools

- **Formatter/Linter**: `ruff` (line-length: 100, target: py310)
- **Testing**: `pytest`
- **Environment**: Conda (Python 3.10)

---

## Project Structure

```
prompt-lora-trainer/
├── configs/                          # Training configurations
│   ├── sft_qwen3_4b.yaml            # Standard SFT config (Video-Diffusion-Prompt-Style)
│   ├── sft_qwen3_4b_deforum.yaml    # Enhanced config for De Forum dataset
│   └── dataset_config.yaml          # Dataset processing configuration
├── scripts/                          # Training and utility scripts
│   ├── train_sft.py                 # Main SFT training script (QLoRA)
│   ├── validate_dataset.py          # Dataset validation utility
│   ├── process_deforum_data.py      # De Forum data processing pipeline
│   ├── upload_dataset_to_hub.py     # HuggingFace Hub upload utility
│   └── merge_and_convert_gguf.py    # LoRA merging and GGUF conversion
├── inputs/                           # Raw input data (storyboard files)
├── outputs/                          # Training outputs (excluded from git)
│   ├── qwen3-4b-prompt-lora/        # LoRA adapter files
│   ├── qwen3-4b-prompt-lora-merged/ # Merged model
│   └── *.gguf                       # Ollama-compatible models
├── notebooks/                        # Jupyter notebooks (empty)
├── wandb/                           # Weights & Biases logs (excluded from git)
├── pyproject.toml                   # Project metadata and dependencies
├── setup_env.sh                     # Environment setup script
├── .env                             # API keys (excluded from git)
├── Modelfile.qwen3-prompt-lora      # Ollama model definition
├── hf-skills-training.md            # HF Skills training reference guide
└── IMPLEMENTATION_PLAN.md           # Detailed implementation plan
```

---

## Build and Development Commands

### Environment Setup

```bash
# Create and configure Conda environment
conda create -n prompt-lora-trainer python=3.10 -y
conda activate prompt-lora-trainer

# Install PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install all dependencies
pip install -e .
# Or install individually:
pip install transformers accelerate datasets peft bitsandbytes trl huggingface-hub wandb python-dotenv pyyaml
```

Or use the setup script:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

### Training Commands

```bash
# Validate dataset format
python scripts/validate_dataset.py --dataset Limbicnation/Video-Diffusion-Prompt-Style

# Dry-run (validate setup without training)
python scripts/train_sft.py --config configs/sft_qwen3_4b.yaml --dry-run

# Full training
python scripts/train_sft.py --config configs/sft_qwen3_4b.yaml

# Override model/dataset via CLI
python scripts/train_sft.py --config configs/sft_qwen3_4b.yaml --model Qwen/Qwen3-8B-Instruct
```

### Dataset Processing (De Forum)

```bash
# Process raw storyboard files into training dataset
python scripts/process_deforum_data.py \
    --input_dir ./inputs \
    --output_file ./data/deforum_prompts_processed.json \
    --augmentation_factor 20

# Upload to HuggingFace Hub
python scripts/upload_dataset_to_hub.py \
    --input_json ./data/deforum_prompts_processed.json \
    --repo_name "Limbicnation/deforum-prompt-lora-dataset"
```

### Post-Training

```bash
# Merge LoRA adapter with base model
python scripts/merge_and_convert_gguf.py \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --lora-adapter Limbicnation/qwen3-4b-prompt-lora \
    --output-dir ./outputs/merged

# Create Ollama model
ollama create qwen3-prompt-lora -f Modelfile.qwen3-prompt-lora
ollama run qwen3-prompt-lora "Generate a cinematic forest video prompt"
```

### Code Quality

```bash
# Run linter
ruff check .

# Run formatter
ruff format .

# Run tests
pytest
```

---

## Configuration System

### Training Config (YAML)

Training configurations are stored in `configs/` as YAML files. Key parameters:

```yaml
# Model
model_id: "Qwen/Qwen3-4B-Instruct-2507"

# Dataset
dataset_id: "Limbicnation/Video-Diffusion-Prompt-Style"
max_seq_length: 512

# LoRA Configuration
lora_r: 16                    # Rank (higher = more capacity)
lora_alpha: 32                # Scaling factor (typically 2x rank)
lora_dropout: 0.05            # Regularization
lora_target_modules:          # Which layers to adapt
  - q_proj
  - k_proj
  - v_proj
  - o_proj

# Quantization (QLoRA)
use_4bit: true
bnb_4bit_compute_dtype: "bfloat16"
bnb_4bit_quant_type: "nf4"

# Training Hyperparameters
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4      # Effective batch = 16
learning_rate: 2.0e-4
warmup_ratio: 0.03
lr_scheduler_type: "cosine"

# Optimizer
optim: "paged_adamw_8bit"     # Memory-efficient for QLoRA
gradient_checkpointing: true   # Trade compute for VRAM
bf16: true                     # Prefer over fp16 on Ampere+

# Output
output_dir: "./outputs/qwen3-4b-prompt-lora"
logging_steps: 10
save_steps: 100
push_to_hub: true
hub_model_id: "Limbicnation/qwen3-4b-prompt-lora"

# Monitoring
report_to: "wandb"
run_name: "sft-qwen3-4b-video-prompts"
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Hugging Face Token (required for dataset/model access)
HF_TOKEN=hf_your_token_here

# Weights & Biases (optional, for training monitoring)
WANDB_API_KEY=your_wandb_key

# Optional: Other API keys
GEMINI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

---

## Code Style Guidelines

### Python Style

- **Line length**: 100 characters (enforced by ruff)
- **Target Python**: 3.10+
- **Import style**: Use `isort` compatible ordering
- **Docstrings**: Google-style docstrings for all public functions

### Linting Rules (ruff)

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
ignore = ["E501"]  # Line length handled by formatter
```

### Code Patterns

```python
#!/usr/bin/env python3
"""
Module docstring with clear description.

Usage:
    python script.py --arg value
"""

import os
import argparse
from typing import List, Dict, Optional
from dataclasses import dataclass

import torch
from dotenv import load_dotenv

# Load environment variables at module level
load_dotenv()


@dataclass
class ConfigClass:
    """Type-safe configuration with defaults."""
    param: str = "default"


def function_with_docstring(arg: str) -> Dict:
    """
    Clear description of function purpose.
    
    Args:
        arg: Description of argument
        
    Returns:
        Description of return value
    """
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--config", type=str, help="Path to config")
    args = parser.parse_args()
    # Main execution
```

---

## Training Architecture

### SFT Training Flow

1. **Load Dataset**: From HuggingFace Hub using `datasets.load_dataset()`
2. **Format Prompts**: Convert dataset rows to instruction-following format
3. **Quantization**: Configure 4-bit NF4 quantization with `BitsAndBytesConfig`
4. **Load Model**: With `AutoModelForCausalLM.from_pretrained()` and quantization
5. **Prepare for Training**: `prepare_model_for_kbit_training()` for QLoRA
6. **Configure LoRA**: `LoraConfig` with target attention modules
7. **Training Arguments**: `SFTConfig` with memory optimizations
8. **Train**: `SFTTrainer` with Weights & Biases monitoring
9. **Save & Push**: Local save + HuggingFace Hub upload

### Prompt Format

Training uses an instruction-following format:

```
### Instruction:
Generate a high-quality video prompt based on the following style.

### Style:
{style_name}

### Response:
{prompt_text}

### Negative Prompt:
{negative_prompt}

### Tags:
{tags}
```

### Memory Optimization Strategy

- **QLoRA**: 4-bit quantization reduces model memory from ~8GB to ~2GB
- **Gradient Checkpointing**: Trade compute for VRAM
- **Paged Optimizer**: `paged_adamw_8bit` offloads optimizer states to CPU
- **Auto Batch Size**: Halves batch size if OOM detected
- **BF16**: Native precision on RTX 4090 for stability

---

## Testing Strategy

### Dataset Validation

```bash
# Validate dataset format before training
python scripts/validate_dataset.py --dataset <dataset_id>
```

Checks for:
- SFT compatibility (messages, text, prompt_response, instruction_output formats)
- DPO compatibility (chosen/rejected columns)
- Sample row display for manual inspection

### Dry-Run Training

```bash
# Validate training setup without full execution
python scripts/train_sft.py --config configs/sft_qwen3_4b.yaml --dry-run
```

Validates:
- Configuration loading
- Dataset accessibility
- Model loading
- Sample prompt formatting

---

## Deployment

### HuggingFace Hub

Models are automatically pushed to Hub with:
- LoRA adapter files (safetensors)
- Training configuration
- Model card with training details

### Ollama / GGUF

For local deployment:

1. Convert to GGUF format using `llama.cpp`:
   ```bash
   python convert_hf_to_gguf.py <merged_model_dir> --outtype q4_k_m
   ```

2. Create Ollama model with `Modelfile`:
   ```dockerfile
   FROM ./outputs/model-Q8_0.gguf
   
   SYSTEM """You are an expert AI Video Prompt Engineer..."""
   
   PARAMETER temperature 0.7
   PARAMETER top_p 0.8
   PARAMETER num_ctx 2048
   ```

---

## Security Considerations

### API Keys

- **NEVER** commit `.env` files (already in `.gitignore`)
- **NEVER** hardcode API tokens in scripts
- Use `os.environ.get("HF_TOKEN")` pattern
- Rotate tokens regularly

### Model Safety

- Models are pushed to private repos by default (configure `private: true`)
- Review generated prompts before use in production
- Be aware of potential bias in training data

---

## Common Tasks for Agents

### Adding a New Training Script

1. Create script in `scripts/` with proper shebang and docstring
2. Use `TrainingConfig` dataclass pattern for configuration
3. Support `--dry-run` flag for validation
4. Add to this AGENTS.md documentation

### Adding a New Dataset

1. Process raw data to required format (instruction/response pairs)
2. Validate with `validate_dataset.py`
3. Upload to Hub with `upload_dataset_to_hub.py`
4. Update training config with new `dataset_id`

### Modifying LoRA Configuration

1. Edit appropriate config in `configs/`
2. Key parameters to adjust:
   - `lora_r`: Higher for more capacity (16-128)
   - `lora_alpha`: Typically 2x rank
   - `lora_target_modules`: Add MLP layers for more adaptation

### Debugging Training Issues

1. Check GPU memory: `nvidia-smi`
2. Run dry-run: `--dry-run` flag
3. Reduce batch size or enable `auto_find_batch_size`
4. Check Weights & Biases logs for loss curves
5. Verify dataset format with `validate_dataset.py`

---

## References

- [Hugging Face TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-4B-Instruct)
- [hf-skills-training.md](./hf-skills-training.md) - Hugging Face Skills reference
