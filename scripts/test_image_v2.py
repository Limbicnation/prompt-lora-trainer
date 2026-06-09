#!/usr/bin/env python3
"""Test the v2 image-prompt adapter on its held-out validation set.

Two checks:
  1. Teacher-forced loss on the Hub `validation` split. Should land near
     the trainer-recorded best (0.6339) — large divergence means the adapter
     didn't load correctly.
  2. Free generation on a handful of validation instructions. Prints outputs
     and runs cheap sanity rules: Negative line present, no token leakage,
     no `--ar`/`--model`/`--seed` artifacts, prompt body in 50-1500 chars.

Usage:
    python scripts/test_image_v2.py
    python scripts/test_image_v2.py --gen-n 5 --max-rows 50
"""
import argparse
import re
import sys
import time

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ADAPTER_DIR = "/home/gero/GitHub/DeepLearning_Lab/prompt-lora-trainer/outputs/qwen2-5-7b-image-prompt-lora-v2"
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET_ID = "Limbicnation/images-diffusion-prompt-style-v2"

ARTIFACT_RE = re.compile(r"--(ar|model|seed|stylize|s|q|niji|chaos|weird)\b", re.IGNORECASE)
LEAK_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]


def load_model(use_4bit: bool):
    bnb = None
    dtype = torch.bfloat16
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        dtype = None

    print(f"📥 Loading base: {BASE_ID} (4bit={use_4bit})")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=bnb,
    )
    print(f"📥 Loading adapter: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


@torch.inference_mode()
def teacher_forced_loss(model, tok, val, max_rows: int):
    rows = val.select(range(min(max_rows, len(val))))
    losses_sum = 0.0
    n_tokens = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        text = row["text"]
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(model.device)
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        n = (labels != -100).sum().item()
        losses_sum += out.loss.item() * n
        n_tokens += n
        if (i + 1) % 25 == 0:
            print(f"   step {i + 1}/{len(rows)}", flush=True)
    elapsed = time.time() - t0
    avg = losses_sum / max(n_tokens, 1)
    return avg, len(rows), elapsed


@torch.inference_mode()
def generate_samples(model, tok, val, n: int):
    samples = val.select(range(min(n, len(val))))
    results = []
    for row in samples:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert image prompt generator for Stable "
                    "Diffusion / FLUX. When given a style name or scene "
                    "description, output the image prompt followed by an "
                    "optional negative prompt on a new line prefixed with "
                    "'Negative:'. No labels, no preamble, no command-line flags."
                ),
            },
            {"role": "user", "content": row["instruction"]},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tok.eos_token_id,
        )
        gen = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        results.append({
            "instruction": row["instruction"],
            "style": row.get("style", ""),
            "axis": row.get("axis", ""),
            "ref_response": row["response"],
            "ref_negative": row["negative_prompt"],
            "gen": gen,
        })
    return results


def quality_checks(gen: str) -> dict:
    body, _, neg = gen.partition("\nNegative:")
    body = body.strip()
    neg = neg.strip()
    return {
        "has_negative": bool(neg),
        "no_artifacts": not ARTIFACT_RE.search(gen),
        "no_token_leak": all(t not in gen for t in LEAK_TOKENS),
        "len_ok": 50 <= len(body) <= 1500,
        "body_len": len(body),
        "negative_len": len(neg),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gen-n", type=int, default=5)
    p.add_argument("--max-rows", type=int, default=100)
    p.add_argument("--no-4bit", action="store_true")
    args = p.parse_args()

    print(f"🌐 Loading dataset: {DATASET_ID}")
    val = load_dataset(DATASET_ID, split="validation")
    print(f"   validation rows: {len(val)}")

    model, tok = load_model(use_4bit=not args.no_4bit)

    print("\n=== Teacher-forced loss ===")
    avg, n, dt = teacher_forced_loss(model, tok, val, args.max_rows)
    print(f"\n   avg_loss = {avg:.4f}   over {n} rows   ({dt:.1f}s)")
    print(f"   trainer best (record): 0.6339   →   delta {avg - 0.6339:+.4f}")
    if abs(avg - 0.6339) > 0.05:
        print("   ⚠️  large gap — verify adapter loaded onto right base")
    else:
        print("   ✅ within +/-0.05 of trainer-recorded loss")

    print("\n=== Held-out generation ===")
    gens = generate_samples(model, tok, val, args.gen_n)
    pass_counts = {"has_negative": 0, "no_artifacts": 0, "no_token_leak": 0, "len_ok": 0}
    for i, g in enumerate(gens, 1):
        q = quality_checks(g["gen"])
        for k in pass_counts:
            if q[k]:
                pass_counts[k] += 1
        print(f"\n--- sample {i} | style={g['style']} | axis={g['axis']} ---")
        print(f"INSTRUCTION: {g['instruction'][:150]}")
        print(f"GEN:\n{g['gen']}")
        print(f"checks: {q}")

    print("\n=== Quality summary ===")
    for k, v in pass_counts.items():
        print(f"   {k:18s} {v}/{len(gens)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
