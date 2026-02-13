#!/usr/bin/env python3
"""
HuggingFace Dataset Upload Script

Converts processed JSON data to HuggingFace Dataset format and uploads to Hub.

Usage:
    python scripts/upload_dataset_to_hub.py \
        --input_json ./data/deforum_prompts_processed.json \
        --repo_name "Limbicnation/deforum-prompt-lora-dataset" \
        --split_ratio 0.9 \
        --private False
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

from datasets import Dataset, DatasetDict, Features, Value, Sequence
from dotenv import load_dotenv

load_dotenv()


# Dataset features schema (flattened for compatibility)
DATASET_FEATURES = Features({
    "instruction": Value("string"),
    "response": Value("string"),
    "style_name": Value("string"),
    "negative_prompt": Value("string"),
    "tags": Sequence(Value("string")),
    "camera_movement": Value("string"),
    "aspect_ratio": Value("string"),
    "model": Value("string"),
    "seed": Value("int32"),
    "guidance_scale": Value("float32"),
    "steps": Value("int32"),
    "scene_context": Value("string"),
    "text": Value("string"),  # Formatted chat text
})


def format_chat_template_qwen3(example: Dict[str, Any]) -> str:
    """Format example with Qwen3-4B chat template."""
    # Extract technical params for formatting (support both flattened and nested)
    if "technical_params" in example:
        tech = example["technical_params"]
    else:
        tech = {
            "aspect_ratio": example.get("aspect_ratio", "16:9"),
            "model": example.get("model", "WanVideo"),
            "seed": example.get("seed", 42)
        }
    tech_str = f"--ar {tech.get('aspect_ratio', '16:9')} --model {tech.get('model', 'WanVideo')} --seed {tech.get('seed', 42)}"
    
    # Format response with technical details
    response = example.get("response", "")
    negative = example.get("negative_prompt", "")
    tags = example.get("tags", [])
    tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
    
    formatted_response = f"""{response}

Technical Parameters:
- Negative Prompt: {negative}
- Tags: {tags_str}
- Settings: {tech_str}"""
    
    # Qwen3 chat template
    text = f"""<|im_start|>user
{example.get('instruction', '')}<|im_end|>
<|im_start|>assistant
{formatted_response}<|im_end|>"""
    
    return text


def format_chat_template_alpaca(example: Dict[str, Any]) -> str:
    """Alternative Alpaca-style format."""
    return f"""### Instruction:
{example.get('instruction', '')}

### Response:
{example.get('response', '')}

### Technical:
Camera: {example.get('camera_movement', '')}
Tags: {', '.join(example.get('tags', []))}"""


def load_processed_data(input_path: str) -> List[Dict[str, Any]]:
    """Load processed JSON data."""
    print(f"📖 Loading data from {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Loaded {len(data)} examples")
    return data


def split_dataset(data: List[Dict], split_ratio: float = 0.9, seed: int = 42) -> tuple:
    """Split data into train and validation sets."""
    import random
    random.seed(seed)
    
    # Shuffle data
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    # Calculate split
    train_size = int(len(shuffled) * split_ratio)
    train_data = shuffled[:train_size]
    val_data = shuffled[train_size:]
    
    print(f"   Split: {len(train_data)} train, {len(val_data)} validation")
    return train_data, val_data


def flatten_technical_params(example: Dict) -> Dict:
    """Flatten technical_params dict into top-level fields."""
    tech = example.get("technical_params", {})
    example["aspect_ratio"] = tech.get("aspect_ratio", "16:9")
    example["model"] = tech.get("model", "WanVideo")
    example["seed"] = tech.get("seed", 42)
    example["guidance_scale"] = tech.get("guidance_scale", 7.5)
    example["steps"] = tech.get("steps", 30)
    # Remove nested dict
    example.pop("technical_params", None)
    return example

def create_hf_dataset(train_data: List[Dict], val_data: List[Dict], chat_format: str = "qwen3") -> DatasetDict:
    """Create HuggingFace Dataset with formatted text."""
    print("\n🔄 Formatting with chat template...")
    
    # Choose formatter
    if chat_format == "qwen3":
        formatter = format_chat_template_qwen3
    else:
        formatter = format_chat_template_alpaca
    
    # Process each example
    for split_name, data in [("train", train_data), ("validation", val_data)]:
        for example in data:
            example["text"] = formatter(example)
            example = flatten_technical_params(example)
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data, features=DATASET_FEATURES)
    val_dataset = Dataset.from_list(val_data, features=DATASET_FEATURES)
    
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    return dataset


def generate_dataset_card(repo_name: str, num_train: int, num_val: int) -> str:
    """Generate README.md content for the dataset."""
    card = f"""---
language:
- en
license: mit
library_name: datasets
tags:
- video-diffusion
- prompt-generation
- lora-training
- deforum
- ai-art
- cinematic
---

# De Forum Cinematic Prompt Dataset

A specialized dataset for fine-tuning language models to generate cinematic video diffusion prompts in the style of "The Deforum Art Film".

## Description

This dataset contains instruction-response pairs for training models to generate high-quality video diffusion prompts with:
- Cinematic language and film terminology
- De Forum aesthetic (noir, minimalist, art film style)
- Technical parameters (aspect ratio, guidance scale, seeds)
- Camera movements and lighting descriptions
- Negative prompts for quality filtering

## Dataset Structure

### Fields

- `instruction`: User request for prompt generation
- `response`: Generated video diffusion prompt with cinematic descriptions
- `style_name`: Scene identifier from the storyboard
- `negative_prompt`: Terms to exclude from generation
- `tags`: Scene categorization tags (noir, cinematic, psychological, etc.)
- `camera_movement`: Recommended camera technique
- `aspect_ratio`: Video aspect ratio (16:9, 2.39:1, etc.)
- `model`: Video diffusion model (WanVideo, CogVideoX, etc.)
- `seed`: Random seed for reproducibility
- `guidance_scale`: CFG scale (7.0-8.5)
- `steps`: Inference steps (25, 30, 50)
- `scene_context`: Additional narrative context
- `text`: Full formatted conversation with Qwen3 chat template

### Example

```json
{{
  "instruction": "Generate a cinematic video prompt for:\\nScene: INT. SARAH'S STUDIO - DAY\\n...",
  "response": "Cinematic art film scene: INT. SARAH'S STUDIO - DAY...",
  "style_name": "INT. SARAH'S STUDIO - DAY",
  "negative_prompt": "blurry, static, distorted, modern elements...",
  "tags": ["noir", "minimalist", "psychological"],
  "camera_movement": "slow tracking shot following subject",
  "technical_params": {{
    "aspect_ratio": "16:9",
    "model": "WanVideo",
    "seed": 4242,
    "guidance_scale": 7.5,
    "steps": 30
  }},
  "scene_context": "Sarah wakes up to find herself surrounded by..."
}}
```

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("{repo_name}")

# Access training split
train_data = dataset["train"]

# Access a sample
sample = train_data[0]
print(sample["instruction"])
print(sample["response"])
```

### For Training with TRL

```python
from trl import SFTTrainer
from datasets import load_dataset

dataset = load_dataset("{repo_name}", split="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",  # Pre-formatted with chat template
    ...
)
```

## Data Source

This dataset is derived from "The Deforum Art Film" project storyboard materials, including:
- Narrative scene descriptions
- JSON frame sequences with timestamps
- Visual style descriptors
- Negative prompt collections

## Intended Use

- Fine-tuning language models for video diffusion prompt generation
- Training models to understand cinematic language and camera terminology
- Creating domain-specific LoRA adapters for the De Forum aesthetic

## Limitations

- Specialized for noir/art film aesthetic - may not generalize to all video styles
- Based on a single narrative work - limited diversity in source material
- Optimized for Qwen3 chat template - other formats may require adaptation

## Citation

```bibtex
@dataset{{{repo_name.split('/')[-1]},
  author = {{Limbicnation}},
  title = {{De Forum Cinematic Prompt Dataset}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo_name}}}}}
}}
```

## License

MIT License - See LICENSE file for details.
"""
    return card


def upload_to_hub(dataset: DatasetDict, repo_name: str, private: bool = False, token: str = None) -> None:
    """Upload dataset to HuggingFace Hub."""
    print(f"\n📤 Uploading to HuggingFace Hub: {repo_name}...")
    
    if token is None:
        token = os.environ.get("HF_TOKEN")
    
    if not token:
        raise ValueError("HF_TOKEN not found. Set HF_TOKEN environment variable.")
    
    # Generate dataset card
    num_train = len(dataset["train"])
    num_val = len(dataset["validation"])
    readme = generate_dataset_card(repo_name, num_train, num_val)
    
    # Upload with dataset card
    dataset.push_to_hub(
        repo_name,
        private=private,
        token=token,
        commit_message="Add De Forum cinematic prompt dataset"
    )
    
    # Upload README separately for better rendering
    from huggingface_hub import HfApi
    api = HfApi()
    
    try:
        api.upload_file(
            path_or_fileobj=readme.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            token=token,
            commit_message="Add dataset card"
        )
    except Exception as e:
        print(f"   ⚠️ Could not upload README: {e}")
    
    visibility = "private" if private else "public"
    print(f"   ✅ Uploaded {visibility} dataset: https://huggingface.co/datasets/{repo_name}")


def verify_dataset(repo_name: str, token: str = None) -> None:
    """Verify the uploaded dataset can be loaded."""
    print("\n🔍 Verifying uploaded dataset...")
    
    from datasets import load_dataset
    
    if token is None:
        token = os.environ.get("HF_TOKEN")
    
    try:
        ds = load_dataset(repo_name, token=token)
        print(f"   ✅ Successfully loaded dataset")
        print(f"   Train: {len(ds['train'])} examples")
        print(f"   Validation: {len(ds['validation'])} examples")
        print(f"   Columns: {list(ds['train'].column_names)}")
        
        # Show sample
        sample = ds['train'][0]
        print(f"\n   Sample instruction (truncated):")
        print(f"   {sample['instruction'][:100]}...")
        
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")


def main():
    parser = argparse.ArgumentParser(description="Upload De Forum dataset to HuggingFace Hub")
    parser.add_argument("--input_json", type=str, required=True, help="Path to processed JSON file")
    parser.add_argument("--repo_name", type=str, default="Limbicnation/deforum-prompt-lora-dataset", help="HF Hub repo name")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/validation split ratio")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    parser.add_argument("--chat_format", type=str, default="qwen3", choices=["qwen3", "alpaca"], help="Chat template format")
    parser.add_argument("--verify", action="store_true", help="Verify dataset after upload")
    args = parser.parse_args()
    
    print("🚀 De Forum Dataset Upload Pipeline")
    print("=" * 50)
    
    # Check for HF token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not set!")
        print("   Run: export HF_TOKEN=your_huggingface_token")
        return
    
    # Load processed data
    data = load_processed_data(args.input_json)
    
    if not data:
        print("❌ No data found!")
        return
    
    # Split data
    print("\n✂️ Splitting dataset...")
    train_data, val_data = split_dataset(data, args.split_ratio)
    
    # Create HF dataset
    dataset = create_hf_dataset(train_data, val_data, args.chat_format)
    
    # Upload to Hub
    upload_to_hub(dataset, args.repo_name, args.private, token)
    
    # Verify if requested
    if args.verify:
        verify_dataset(args.repo_name, token)
    
    print("\n🎉 Upload complete!")
    print(f"   Dataset URL: https://huggingface.co/datasets/{args.repo_name}")


if __name__ == "__main__":
    main()
