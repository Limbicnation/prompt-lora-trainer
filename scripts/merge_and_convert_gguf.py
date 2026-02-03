#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and convert to GGUF for Ollama.

Usage:
    python scripts/merge_and_convert_gguf.py --output-dir ./outputs/merged

Requirements:
    pip install llama-cpp-python
    # Or use llama.cpp directly for conversion
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and convert to GGUF")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507", help="Base model ID")
    parser.add_argument("--lora-adapter", default="Limbicnation/qwen3-4b-prompt-lora", help="LoRA adapter ID")
    parser.add_argument("--output-dir", default="./outputs/merged", help="Output directory for merged model")
    parser.add_argument("--gguf-output", default="./outputs/qwen3-4b-prompt-lora.gguf", help="GGUF output path")
    parser.add_argument("--quantization", default="q4_k_m", help="GGUF quantization type")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load base model and LoRA
    print(f"📦 Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Use CPU for merging to save VRAM
        trust_remote_code=True,
    )
    
    print(f"🔗 Loading LoRA adapter: {args.lora_adapter}")
    model = PeftModel.from_pretrained(base_model, args.lora_adapter)
    
    # Step 2: Merge LoRA into base model
    print("🔀 Merging LoRA into base model...")
    merged_model = model.merge_and_unload()
    
    # Step 3: Save merged model
    print(f"💾 Saving merged model to: {args.output_dir}")
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✅ Merged model saved to: {args.output_dir}")
    print()
    print("📋 Next steps for GGUF conversion:")
    print()
    print("Option 1: Use llama.cpp (recommended)")
    print(f"  git clone https://github.com/ggerganov/llama.cpp")
    print(f"  cd llama.cpp && make")
    print(f"  python convert_hf_to_gguf.py {args.output_dir} --outtype {args.quantization} --outfile {args.gguf_output}")
    print()
    print("Option 2: Use HuggingFace transformers (if gguf-py installed)")
    print(f"  pip install gguf")
    print(f"  python -m gguf.convert {args.output_dir} {args.gguf_output}")
    print()
    print("Option 3: Create Ollama model directly with GGUF")
    print("  ollama create qwen3-4b-prompt-lora -f Modelfile.limbicnation")
    print()

if __name__ == "__main__":
    main()
