#!/usr/bin/env python3
"""Smoke-test the dual-stream prompt adapter: one generation per dialect.

Loads Qwen2.5-7B (4-bit) + the dual-stream LoRA and generates a prompt for each
target dialect using the same instruction prefixes the model was trained on.
Runs cheap sanity rules (no chat-token leakage, no CLI flags, length window).

Usage:
    python scripts/test_dual_stream.py
    python scripts/test_dual_stream.py --concept "a lighthouse on a storm-battered cliff at dawn"
"""
import argparse
import re
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ADAPTER_DIR = (
    "/home/gero/GitHub/DeepLearning_Lab/prompt-lora-trainer/"
    "outputs/qwen2-5-7b-dual-stream-prompt-lora"
)
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM = (
    "You are an expert image prompt generator for Stable Diffusion / FLUX. "
    "Given a concept and a target model, produce the prompt in that model's dialect. "
    "No labels, no preamble, no command-line flags."
)

DIALECT_PREFIXES = {
    "flux_t5": "Generate a FLUX (T5-XXL) image prompt for:",
    "sdxl_dual_clip": "Generate an SDXL image prompt for:",
    "compact_caption": "Write a compact descriptive caption for:",
    "steering_modifiers": "List image steering modifiers for:",
}

LEAK_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
ARTIFACT_RE = re.compile(r"--(ar|model|seed|stylize|niji|chaos)\b", re.IGNORECASE)


def load():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"📥 Loading base: {BASE_ID} (4bit)")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, device_map="auto", quantization_config=bnb
    )
    print(f"📥 Loading adapter: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


@torch.inference_mode()
def generate(model, tok, instruction: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instruction},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=220,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def checks(text: str) -> str:
    flags = []
    if any(t in text for t in LEAK_TOKENS):
        flags.append("TOKEN_LEAK")
    if ARTIFACT_RE.search(text):
        flags.append("CLI_FLAG")
    if not (10 <= len(text) <= 1500):
        flags.append(f"LEN={len(text)}")
    return "✓ clean" if not flags else "⚠ " + ", ".join(flags)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--concept",
        default="a lighthouse on a storm-battered cliff at dawn",
        help="A novel concept (out-of-distribution tests generalization)",
    )
    args = p.parse_args()

    model, tok = load()
    print(f"\n🎬 Concept: {args.concept!r}\n" + "=" * 70)
    for dialect, prefix in DIALECT_PREFIXES.items():
        instruction = f"{prefix} {args.concept}"
        gen = generate(model, tok, instruction)
        words = len(gen.split())
        print(f"\n### {dialect}  [{words}w]  {checks(gen)}")
        print(f"  prompt: {instruction}")
        print(f"  → {gen}")
    print("\n" + "=" * 70 + "\n✅ Smoke test complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
