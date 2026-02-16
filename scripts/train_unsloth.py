#!/usr/bin/env python3
"""
Unsloth Training Script for Qwen3 LoRA
Optimized for fast training on RTX 4090.
"""

import os
import argparse
import yaml
from dataclasses import dataclass, field
from typing import Optional

from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TrainingConfig:
    """Training configuration."""
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    dataset_id: str = "Limbicnation/deforum-prompt-lora-dataset-v2"
    max_seq_length: int = 512
    
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    fp16: bool = False
    bf16: bool = True
    
    output_dir: str = "./outputs/qwen3-4b-deforum-prompt-lora-v2"
    logging_steps: int = 10
    save_steps: int = 100
    push_to_hub: bool = True
    hub_model_id: Optional[str] = None
    
    report_to: str = "wandb"
    run_name: Optional[str] = None


def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    # Filter only valid fields for TrainingConfig
    valid_fields = {f.name for f in TrainingConfig.__dataclass_fields__.values()}
    filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
    return TrainingConfig(**filtered_dict)


def format_prompt(example: dict) -> str:
    """Format dataset row into training prompt."""
    instruction = example.get("instruction", "Generate a cinematic video prompt.")
    response = example.get("response", "")
    style_name = example.get("style_name", "")
    
    return f"""### Instruction:
{instruction}

### Style:
{style_name}

### Response:
{response}
""".strip()


def main():
    parser = argparse.ArgumentParser(description="Unsloth LoRA Training")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without training")
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = TrainingConfig()
    
    print(f"🚀 Unsloth Training Pipeline")
    print(f"   Model: {config.model_id}")
    print(f"   Dataset: {config.dataset_id}")
    print(f"   Output: {config.output_dir}")
    print()
    
    token = os.environ.get("HF_TOKEN")
    
    # Load dataset
    print(f"📊 Loading dataset: {config.dataset_id}...")
    dataset = load_dataset(config.dataset_id, split="train", token=token)
    print(f"   Rows: {len(dataset)}")
    
    # Format dataset
    print("🔄 Formatting prompts...")
    dataset = dataset.map(lambda x: {"text": format_prompt(x)}, remove_columns=dataset.column_names)
    
    # Split train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"   Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    if args.dry_run:
        print("\n[DRY RUN] Sample prompt:")
        print("-" * 50)
        print(train_dataset[0]["text"][:500])
        print("-" * 50)
        return
    
    # Load model with unsloth (4-bit quantization built-in)
    print(f"📦 Loading model with unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_id,
        max_seq_length=config.max_seq_length,
        dtype="float16",  # Use float16 instead of bfloat16
        load_in_4bit=True,
        token=token,
    )
    
    # Add LoRA adapters
    print("🔗 Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Optimized checkpointing
        random_state=3407,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        output_dir=config.output_dir,
        seed=3407,
        report_to=config.report_to,
        run_name=config.run_name,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
    )
    
    # Trainer
    print("🏋️ Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=2,
        args=training_args,
    )
    
    trainer.train()
    
    # Save
    print(f"💾 Saving model...")
    model.save_pretrained_merged(
        config.output_dir,
        tokenizer,
        save_method="merged_16bit",
    )
    
    if config.push_to_hub:
        print(f"📤 Pushing to Hub...")
        model.push_to_hub_merged(
            config.hub_model_id or f"Limbicnation/qwen3-4b-deforum-prompt-lora-v2",
            tokenizer,
            save_method="merged_16bit",
        )
    
    print("🎉 Training complete!")


if __name__ == "__main__":
    main()
