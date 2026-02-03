#!/usr/bin/env python3
"""
Dataset Validation Script

Validates that a HuggingFace dataset is properly formatted for SFT training.
Based on hf-skills-training.md dataset validation guidance.

Usage:
    python scripts/validate_dataset.py --dataset Limbicnation/Video-Diffusion-Prompt-Style
"""

import argparse
import os
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


def validate_sft_columns(dataset) -> dict:
    """Check if dataset has required SFT columns."""
    columns = dataset.column_names
    
    # SFT can work with various formats
    valid_formats = {
        "messages": "messages" in columns,  # Chat format
        "text": "text" in columns,  # Plain text
        "prompt_response": "prompt" in columns and "response" in columns,
        "instruction_output": "instruction" in columns and "output" in columns,
        "prompt_text": "prompt_text" in columns,  # Our custom format
    }
    
    return valid_formats


def validate_dpo_columns(dataset) -> bool:
    """Check if dataset has required DPO columns."""
    columns = dataset.column_names
    return "chosen" in columns and "rejected" in columns


def main():
    parser = argparse.ArgumentParser(description="Validate dataset for training")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset ID")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    args = parser.parse_args()
    
    token = os.environ.get("HF_TOKEN")
    
    print(f"📊 Validating dataset: {args.dataset}")
    print(f"   Split: {args.split}")
    print()
    
    try:
        dataset = load_dataset(args.dataset, split=args.split, token=token)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    print(f"✅ Dataset loaded successfully")
    print(f"   Rows: {len(dataset)}")
    print(f"   Columns: {dataset.column_names}")
    print()
    
    # Sample
    print("📝 Sample row:")
    print("-" * 50)
    sample = dataset[0]
    for key, value in sample.items():
        val_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
        print(f"  {key}: {val_str}")
    print("-" * 50)
    print()
    
    # SFT validation
    print("🔍 SFT Compatibility:")
    sft_formats = validate_sft_columns(dataset)
    sft_ready = any(sft_formats.values())
    
    for format_name, is_valid in sft_formats.items():
        status = "✓" if is_valid else "✗"
        print(f"   {status} {format_name}")
    
    if sft_ready:
        print(f"\n   ✅ SFT: READY")
    else:
        print(f"\n   ❌ SFT: INCOMPATIBLE")
    
    # DPO validation
    print("\n🔍 DPO Compatibility:")
    dpo_ready = validate_dpo_columns(dataset)
    if dpo_ready:
        print("   ✓ chosen column")
        print("   ✓ rejected column")
        print(f"\n   ✅ DPO: READY")
    else:
        print("   ✗ Missing 'chosen' and/or 'rejected' columns")
        print(f"\n   ❌ DPO: INCOMPATIBLE")
    
    print()


if __name__ == "__main__":
    main()
