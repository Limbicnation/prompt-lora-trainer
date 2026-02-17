#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) Script for Prompt Generation LoRA

Based on hf-skills-training.md best practices for LoRA/QLoRA training.
Optimized for Qwen3-4B/8B on RTX 4090 (24GB VRAM).

Usage:
    python scripts/train_sft.py --config configs/sft_qwen3_4b.yaml
    python scripts/train_sft.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset Limbicnation/Video-Diffusion-Prompt-Style
"""

import os
import argparse
import yaml
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
import importlib

load_dotenv()

# Optional trackio import
trackio = None
try:
    import trackio as _trackio
    trackio = _trackio
except ImportError:
    pass


@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults for consumer hardware."""
    
    # Model
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    
    # Dataset
    dataset_id: str = "Limbicnation/Video-Diffusion-Prompt-Style"
    dataset_text_field: str = "prompt_text"
    max_seq_length: int = 512
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Quantization (QLoRA)
    use_4bit: bool = True
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    
    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Optimization
    optim: str = "paged_adamw_8bit"  # More memory efficient for LoRA
    fp16: bool = False
    bf16: bool = True  # Will be auto-detected if not supported
    gradient_checkpointing: bool = True
    auto_find_batch_size: bool = True  # Auto-reduce if OOM
    
    # Packing (memory efficient)
    packing: bool = True
    packing_strategy: str = "wrapped"
    
    # Output
    output_dir: str = "./outputs/qwen3-4b-prompt-lora"
    logging_steps: int = 10
    save_steps: int = 100
    push_to_hub: bool = True
    hub_model_id: Optional[str] = None
    
    # Monitoring
    report_to: str = "wandb"
    run_name: Optional[str] = None
    
    # Evaluation (optional)
    eval_steps: Optional[int] = None
    eval_strategy: Optional[str] = None
    load_best_model_at_end: bool = False
    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None
    early_stopping_patience: Optional[int] = None

    # Preprocessing (optional, for config compatibility)
    preprocessing: Optional[dict] = None


def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict)


def format_prompt(example: dict) -> str:
    """Format dataset row into training prompt with schema auto-detection."""
    # Schema detection
    is_deforum = "instruction" in example and "response" in example
    
    if is_deforum:
        instruction = example.get("instruction", "Generate a cinematic video prompt.")
        response = example.get("response", "")
        style_name = example.get("style_name", "")
        negative = example.get("negative_prompt", "")
        camera = example.get("camera_movement", "")
        tags = ", ".join(example.get("tags", [])) if isinstance(example.get("tags"), list) else ""
        scene = example.get("scene_context", "")
        
        return f"""### Instruction:
{instruction}

### Style:
{style_name}

### Response:
{response}

### Negative Prompt:
{negative}

### Camera:
{camera}

### Tags:
{tags}

### Context:
{scene}""".strip()
    else:
        # Original Video-Diffusion-Prompt-Style format
        instruction = "Generate a high-quality video prompt based on the following style."
        style_name = example.get("style_name", "Cinematic")
        prompt_text = example.get("prompt_text", "")
        negative = example.get("negative_prompt", "")
        tags = ", ".join(example.get("tags", [])) if isinstance(example.get("tags"), list) else ""
        
        return f"""### Instruction:
{instruction}

### Style:
{style_name}

### Response:
{prompt_text}

### Negative Prompt:
{negative}

### Tags:
{tags}""".strip()


def main():
    parser = argparse.ArgumentParser(description="SFT LoRA Training for Prompt Generation")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model", type=str, help="Model ID (overrides config)")
    parser.add_argument("--dataset", type=str, help="Dataset ID (overrides config)")
    parser.add_argument("--output-dir", type=str, help="Output directory (overrides config)")
    parser.add_argument("--dry-run", "--dry_run", action="store_true", help="Validate setup without training")
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = TrainingConfig()
    
    # Override with CLI args
    if args.model:
        config.model_id = args.model
    if args.dataset:
        config.dataset_id = args.dataset
    if args.output_dir:
        config.output_dir = args.output_dir
    
    print(f"🚀 SFT LoRA Training Pipeline")
    print(f"   Model: {config.model_id}")
    print(f"   Dataset: {config.dataset_id}")
    print(f"   Output: {config.output_dir}")
    print()
    
    # Token
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not set. Run: export HF_TOKEN=your_token")
    
    # Load dataset
    print(f"📊 Loading dataset: {config.dataset_id}...")
    try:
        if os.path.exists(config.dataset_id) and config.dataset_id.endswith(".json"):
            dataset = load_dataset("json", data_files=config.dataset_id, split="train")
        else:
            dataset = load_dataset(config.dataset_id, split="train", token=token)
    except Exception as e:
        print(f"⚠️ Hub loading failed: {e}")
        # Try local fallback if not already tried
        local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "deforum_prompts_processed.json")
        if os.path.exists(local_path):
            print(f"📂 Falling back to local data: {local_path}")
            dataset = load_dataset("json", data_files=local_path, split="train")
        else:
            raise e
            
    print(f"   Rows: {len(dataset)}")
    
    # Format dataset: Use existing 'text' or run format_prompt
    if "text" in dataset.column_names:
        print("📝 Using existing 'text' column from dataset.")
        # Ensure we only keep the 'text' column for SFTTrainer
        dataset = dataset.map(lambda x: {"text": x["text"]}, remove_columns=dataset.column_names)
    else:
        print("🔄 Formatting prompts using detected schema...")
        dataset = dataset.map(lambda x: {"text": format_prompt(x)}, remove_columns=dataset.column_names)
    
    # Split into train/eval if eval_strategy is configured
    eval_dataset = None
    if config.eval_strategy:
        split = dataset.train_test_split(test_size=0.1, seed=42)
        dataset = split["train"]
        eval_dataset = split["test"]
        print(f"   Train: {len(dataset)}, Eval: {len(eval_dataset)}")

    if args.dry_run:
        print("\n[DRY RUN] Sample formatted prompt:")
        print("-" * 50)
        print(dataset[0]["text"][:500])
        print("-" * 50)
        print(f"\n✅ Dry run complete. Max Seq Length target: {config.max_seq_length}")
        return
    
    # Quantization config (QLoRA)
    bnb_config = None
    if config.use_4bit:
        print("🔧 Configuring 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )
    elif config.use_8bit:
        print("🔧 Configuring 8-bit quantization...")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load model
    print(f"📦 Loading model: {config.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,  # Required for Qwen models; only use with trusted model sources
        token=token,
    )
    
    # Prepare for k-bit training
    if config.use_4bit or config.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA config
    print("🔗 Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Detect bf16 support
    bf16_supported = torch.cuda.is_bf16_supported(including_emulation=False)
    use_bf16 = config.bf16 and bf16_supported
    
    # Training arguments (following HF SFTConfig best practices)
    training_args = SFTConfig(
        # GROUP 1: Memory usage
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # For newer PyTorch
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        auto_find_batch_size=config.auto_find_batch_size,  # Halves batch if OOM
        
        # GROUP 2: Dataset-related
        dataset_text_field="text",  # Use pre-mapped text field
        max_length=config.max_seq_length,
        packing=config.packing,
        
        # GROUP 3: Training parameters
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,  # paged_adamw_8bit for LoRA
        
        # GROUP 4: Logging/Output
        output_dir=config.output_dir,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        report_to=config.report_to,
        run_name=config.run_name or f"sft-{config.model_id.split('/')[-1]}",
        
        # Precision
        fp16=config.fp16,
        bf16=use_bf16,
        
        # Evaluation
        eval_strategy=config.eval_strategy or "no",
        eval_steps=config.eval_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,

        # Hub
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        hub_token=token,
    )
    
    # Initialize trackio if requested
    if config.report_to == "trackio" and trackio is not None:
        print("📊 Initializing Trackio monitoring...")
        trackio.init(
            run_name=config.run_name or f"sft-{config.model_id.split('/')[-1]}",
            config={
                "model_id": config.model_id,
                "dataset_id": config.dataset_id,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "learning_rate": config.learning_rate,
                "num_train_epochs": config.num_train_epochs,
            }
        )

    # Callbacks
    callbacks = []
    if config.early_stopping_patience and eval_dataset is not None:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))

    # Trainer
    print("🏋️ Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks if callbacks else None,
    )
    
    trainer.train()
    
    # Save
    print(f"💾 Saving model to {config.output_dir}...")
    trainer.save_model()
    
    if config.push_to_hub:
        hub_id = config.hub_model_id or f"Limbicnation/{config.model_id.split('/')[-1]}-prompt-lora"
        print(f"📤 Pushing to Hub: {hub_id}...")
        trainer.push_to_hub()
    
    print("🎉 Training complete!")


if __name__ == "__main__":
    main()
