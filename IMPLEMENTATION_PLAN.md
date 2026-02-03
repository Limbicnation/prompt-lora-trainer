# LoRA Fine-Tuning Implementation Plan

> **Target:** Train a prompt-generation LoRA on Qwen3-4B/8B using the 752-row Video-Diffusion-Prompt-Style dataset.  
> **Hardware:** RTX 4090 (24GB VRAM)  
> **Framework:** TRL + PEFT + bitsandbytes (QLoRA)

---

## 1. Data Preparation

### 1.1 Dataset Loading

```python
from datasets import load_dataset

dataset = load_dataset(
    "Limbicnation/Video-Diffusion-Prompt-Style",
    split="train",
    token=os.environ["HF_TOKEN"]
)
```

### 1.2 Schema

| Column | Type | Description |
|--------|------|-------------|
| `style_name` | str | Style category (e.g., "Cinematic Action") |
| `prompt_text` | str | Main video prompt with flags |
| `negative_prompt` | str | Elements to avoid |
| `tags` | list[str] | Style/category tags |
| `compatible_models` | list[str] | Target models (WanVideo, LTX) |

### 1.3 Tokenization Format

```python
def format_for_sft(example):
    return f"""### Instruction:
Generate a high-quality video prompt.

### Style:
{example['style_name']}

### Response:
{example['prompt_text']}

### Negative:
{example['negative_prompt']}

### Tags:
{', '.join(example['tags'])}"""
```

### 1.4 Validation Script

```bash
python scripts/validate_dataset.py --dataset Limbicnation/Video-Diffusion-Prompt-Style
```

---

## 2. Model Configuration

### 2.1 Base Model Selection

| Model | Parameters | VRAM (QLoRA) | Recommendation |
|-------|------------|--------------|----------------|
| Qwen3-4B-Instruct | 4B | ~8GB | ✅ Primary target |
| Qwen3-8B-Instruct | 8B | ~12GB | For higher quality |

### 2.2 LoRA Hyperparameters

```yaml
lora_r: 16          # Rank (lower = smaller adapter)
lora_alpha: 32      # Scaling factor (typically 2x rank)
lora_dropout: 0.05  # Regularization
target_modules:     # Attention layers only
  - q_proj
  - k_proj
  - v_proj
  - o_proj
```

**Trainable Parameters:** ~0.5% of base model

### 2.3 Quantization (QLoRA)

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

---

## 3. Training Pipeline

### 3.1 Optimizer & Scheduler

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | `paged_adamw_32bit` | Memory-efficient for QLoRA |
| Learning Rate | `2e-4` | Standard for LoRA |
| Scheduler | `cosine` | Smooth decay |
| Warmup | 3% of steps | Gradual ramp-up |

### 3.2 Batch Configuration

```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
# Effective batch size: 16
```

### 3.3 Training Duration

| Metric | Value |
|--------|-------|
| Epochs | 3 |
| Dataset Size | 752 |
| Steps/Epoch | 47 (with batch 16) |
| Total Steps | ~141 |
| Est. Time | 10-15 min (RTX 4090) |

### 3.4 Memory Optimization

```yaml
gradient_checkpointing: true  # Trade compute for VRAM
bf16: true                     # RTX 4090 native precision
fp16: false                    # Prefer bf16 for stability
```

---

## 4. Evaluation & Checkpointing

### 4.1 Monitoring

```yaml
report_to: "wandb"
logging_steps: 10
run_name: "sft-qwen3-4b-video-prompts"
```

**Tracked Metrics:**

- `train/loss`
- `train/learning_rate`
- `train/grad_norm`

### 4.2 Checkpointing Strategy

```yaml
save_steps: 100         # Save every 100 steps
save_total_limit: 3     # Keep last 3 checkpoints
output_dir: "./outputs/qwen3-4b-prompt-lora"
```

### 4.3 Final Validation

```bash
# 1. Dry-run first
python scripts/train_sft.py --config configs/sft_qwen3_4b.yaml --dry-run

# 2. Inspect sample output
# The dry-run prints a formatted sample prompt

# 3. Full training
python scripts/train_sft.py --config configs/sft_qwen3_4b.yaml
```

### 4.4 Hub Push

```yaml
push_to_hub: true
hub_model_id: "Limbicnation/qwen3-4b-prompt-lora"
```

---

## 5. Execution Checklist

- [ ] Validate dataset format
- [ ] Run dry-run training
- [ ] Execute full SFT training
- [ ] Monitor W&B dashboard
- [ ] Push LoRA adapters to Hub
- [ ] Test inference with merged model

---

## 6. Post-Training (Optional)

### 6.1 GGUF Conversion

```bash
# For Ollama/llama.cpp deployment
python -m llama_cpp.convert --outfile model.gguf
```

### 6.2 Inference Test

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct")
model = PeftModel.from_pretrained(base, "Limbicnation/qwen3-4b-prompt-lora")
```
