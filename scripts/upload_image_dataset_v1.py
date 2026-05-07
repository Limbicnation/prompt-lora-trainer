#!/usr/bin/env python3
"""Upload pre-cleaned image-prompt dataset to Hugging Face Hub.

Reads the JSONL produced by the external `merge_and_clean.py`
(/home/gero/GitHub/limbicnation/ComfyUI-PromptGenerator/scripts/merge_and_clean.py),
pre-renders the `text` field with Qwen2.5's chat template (overriding the default
Qwen system message with our prompt-engineering one), splits 90/10, and
pushes to Limbicnation/images-diffusion-prompt-style-v1.

The negative prompt is appended to the assistant turn separated by `\\n\\nNegative: `
so a single-turn generation can include both. Downstream parses by the first
occurrence of `\\nNegative:` at line start.

Usage:
    python scripts/upload_image_dataset_v1.py --dry-run
    python scripts/upload_image_dataset_v1.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

DEFAULT_SOURCE = (
    "/home/gero/GitHub/limbicnation/ComfyUI-PromptGenerator/data/prompts_clean.jsonl"
)
DEFAULT_HUB_ID = "Limbicnation/images-diffusion-prompt-style-v1"
TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42
EVAL_RATIO = 0.10
REQUIRED_KEYS = ("instruction", "response", "negative_prompt")

SYSTEM_PROMPT = (
    "You are an expert image prompt generator for Stable Diffusion / FLUX. "
    "When given a style name or scene description, output the image prompt followed "
    "by an optional negative prompt on a new line prefixed with 'Negative:'. "
    "No labels, no preamble, no command-line flags."
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} invalid JSON: {e}") from e
    return rows


def validate(rows: list[dict]) -> None:
    for i, row in enumerate(rows):
        for key in REQUIRED_KEYS:
            if key not in row:
                raise ValueError(f"row {i} missing key: {key}")
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"row {i} key {key!r} is empty or non-string")


def render_text(tokenizer, row: dict) -> str:
    """Render Qwen chat template with our system prompt and negative appended to assistant turn."""
    assistant_content = f"{row['response']}\n\nNegative: {row['negative_prompt']}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["instruction"]},
        {"role": "assistant", "content": assistant_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--hub-id", default=DEFAULT_HUB_ID)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--private", action="store_true", default=True)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: HF_TOKEN not set. Use --dry-run to skip Hub push.", file=sys.stderr)
        return 1

    if not args.source.exists():
        print(f"ERROR: source not found: {args.source}", file=sys.stderr)
        return 1

    print(f"📂 Loading {args.source}")
    rows = load_jsonl(args.source)
    print(f"   {len(rows)} rows")

    print("🔍 Validating schema")
    validate(rows)

    print(f"🔡 Loading tokenizer: {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, token=token)

    print(f"📝 Rendering text field with chat template ({TOKENIZER_ID})")
    enriched = []
    for row in rows:
        enriched.append(
            {
                "instruction": row["instruction"],
                "response": row["response"],
                "negative_prompt": row["negative_prompt"],
                "text": render_text(tokenizer, row),
            }
        )

    ds = Dataset.from_list(enriched)
    split = ds.train_test_split(test_size=EVAL_RATIO, seed=SEED)
    dsdict = DatasetDict({"train": split["train"], "validation": split["test"]})

    print(f"   train: {len(dsdict['train'])}  validation: {len(dsdict['validation'])}")
    print("🔎 Sample text field (first 600 chars):")
    print("-" * 60)
    print(dsdict["train"][0]["text"][:600])
    print("-" * 60)

    if args.dry_run:
        print("✅ Dry run complete (no Hub push).")
        return 0

    print(f"📤 Pushing to {args.hub_id} (private={args.private})")
    try:
        dsdict.push_to_hub(args.hub_id, private=args.private, token=token)
        print(f"🎉 Done: https://huggingface.co/datasets/{args.hub_id}")
    except Exception as e:
        fallback = Path(__file__).resolve().parent.parent / "data" / "images_diffusion_v1.jsonl"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        with fallback.open("w") as f:
            for row in enriched:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"⚠️  Hub push failed ({e}); wrote local fallback to {fallback}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
