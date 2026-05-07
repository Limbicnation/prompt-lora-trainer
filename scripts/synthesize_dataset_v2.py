#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "google-genai>=0.3.0",
#   "datasets>=2.18.0",
#   "sentence-transformers>=3.0.0",
#   "scikit-learn>=1.3.0",
#   "huggingface-hub>=0.22.0",
#   "tqdm>=4.65.0",
#   "python-dotenv>=1.0.0",
#   "transformers>=4.40.0",
# ]
# ///
"""Synthesize ~5k high-quality image-prompt rows for v2 dataset.

Pipeline:
  Stage 0: Load v1 styles → build seeds (style × axis × N generations)
  Stage 1: Generate (Gemini, parallel ThreadPool)
  Stage 2: Rule-based filter (length, regex, format)
  Stage 3: Semantic dedup (sentence-transformers + cosine)
  Stage 4: LLM-as-judge (Gemini, separate system prompt)
  Stage 5: Push to Limbicnation/images-diffusion-prompt-style-v2

Cost estimate (gemini-2.5-flash, 5k rows): ~$3-6 incl. judge.
Wall time: ~30 min batched + ~2 min embed/dedup.

Usage:
    # Smoke test (3 seeds × 2 = 6 generations, no upload)
    uv run scripts/synthesize_dataset_v2.py --dry-run --max-seeds 3

    # Full run
    uv run scripts/synthesize_dataset_v2.py

    # Override model
    GEMINI_MODEL=gemini-3.1-flash uv run scripts/synthesize_dataset_v2.py
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from huggingface_hub import HfApi
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

load_dotenv()


# =============================================================================
# CONFIG
# =============================================================================

V1_DATASET_ID = "Limbicnation/images-diffusion-prompt-style-v1"
V2_DATASET_ID = "Limbicnation/images-diffusion-prompt-style-v2"
LOCAL_V1_JSONL = "/home/gero/GitHub/limbicnation/ComfyUI-PromptGenerator/data/prompts_clean.jsonl"
TRAIN_TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"   # for text-field rendering

SYSTEM_PROMPT_TRAIN = (
    "You are an expert image prompt generator for Stable Diffusion / FLUX. "
    "When given a style name or scene description, output the image prompt followed "
    "by an optional negative prompt on a new line prefixed with 'Negative:'. "
    "No labels, no preamble, no command-line flags."
)

# Default to gemini-2.5-flash-lite — ~6x cheaper output tokens than flash,
# fits the $5 budget comfortably with thinking ON. Override via env var.
GEN_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
JUDGE_MODEL = os.environ.get("GEMINI_JUDGE_MODEL", GEN_MODEL)

# 2 generations per (style, axis) yields ~5,500 final rows after attrition
# from 451 styles × 8 axes × 2 = 7,216 candidates (~$2-3 with flash-lite).
GENERATIONS_PER_SEED = 2
DEDUP_THRESHOLD = 0.92
MIN_AVG_JUDGE_SCORE = 3.5
MIN_DIM_JUDGE_SCORE = 3
MAX_WORDS = 200
MIN_WORDS = 30
MAX_CONCURRENT = 16
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42

# Diversification axes — each style is expanded across these dimensions
AXES = [
    "dramatic chiaroscuro lighting",
    "minimalist negative-space composition",
    "rich saturated color palette",
    "moody atmospheric depth",
    "intricate hyperdetailed textures",
    "ethereal soft volumetric light",
    "bold graphic geometric structure",
    "weathered organic surface detail",
]

# Forbidden patterns in generated prompts
BAD_PATTERNS = [
    r"--ar\s+\d", r"--seed\s+\d", r"--model\s+\w+", r"--steps\s+\d",
    r"--cfg\s+\d", r"--sampler\s+\w+",
    r"<lora:[^>]+>", r"<embedding:[^>]+>",
    r"```", r"^#{1,6}\s",
    r"^\s*(Here is|Here's|Sure,|I'll generate|Certainly|Of course)",
    r"^\s*\*\*",  # Markdown bold preamble
]

# Curated diverse negative prompts (sampled deterministically by seed hash)
CURATED_NEGATIVES = [
    "blurry, low quality, watermark, deformed, distorted, jpeg artifacts",
    "out of focus, soft edges, low resolution, pixelated, oversharpened",
    "ugly, malformed, anatomically incorrect, disfigured, asymmetric features",
    "text, watermark, signature, logo, frame, border, cropped",
    "noisy, grainy, low contrast, washed out colors, faded",
    "amateurish, sketch, draft, unfinished, low effort",
    "duplicate, multiple views, collage, montage",
    "stock photo, generic, cliche, boring, uninteresting",
    "cartoonish when not requested, plastic skin, dead eyes, vacant expression",
    "overprocessed, oversaturated, hdr glow, lens flare overuse",
    "low quality, deformed, blurry, ugly, bad anatomy, watermark, signature",
    "bad composition, awkward framing, cluttered background, distracting elements",
    "low detail, flat lighting, dull colors, no atmosphere",
    "extra limbs, missing limbs, fused fingers, malformed hands",
    "color banding, posterization, jpeg compression, pixel artifacts",
]


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_GEN = """You are an expert image-prompt engineer for Stable Diffusion / FLUX / Midjourney.

Your task: given a STYLE NAME and a creative AXIS, write ONE image prompt that:
  - Is 30-200 words
  - Uses comma-separated descriptive phrases (not full sentences)
  - Has rich vocabulary — specific lighting, composition, materials, mood
  - Names artists, art movements, or technical references when relevant
  - Does NOT include CLI flags (--ar, --seed, --model, etc.)
  - Does NOT include LoRA tags (<lora:...>)
  - Does NOT include preamble like "Here is" or "Sure"
  - Does NOT include markdown formatting

Output STRICT JSON with exactly these two keys:
{
  "response": "the image prompt",
  "negative_prompt": "comma-separated negative descriptors"
}

No other text. No markdown fences. Just the JSON object."""

SYSTEM_JUDGE = """You are an expert evaluator of image-generation prompts.

Rate the given prompt on FOUR dimensions (1-5 scale, integers only):
  1. vocabulary_richness  - density of specific descriptive lexicon
  2. style_alignment      - matches the requested style name
  3. compositional_clarity - coherent visual scene the model can render
  4. format_adherence     - comma-separated, no flags, no preamble

Output STRICT JSON with exactly these four integer keys:
{"vocabulary_richness": N, "style_alignment": N, "compositional_clarity": N, "format_adherence": N}

No other text. No markdown."""


# =============================================================================
# STAGE 0: SEEDS
# =============================================================================

def _is_clean_style(s: str) -> bool:
    """Reject multiline / markdown / overly-long extractions."""
    if not s or "\n" in s or len(s) > 80 or len(s) < 3:
        return False
    if s.startswith("#") or s.startswith("**") or "```" in s:
        return False
    if any(bad in s.lower() for bad in ("certainly", "sure,", "here is", "namin")):
        return False
    return True


def extract_styles(dataset: Dataset) -> list[str]:
    """Extract clean unique style names from v1 instructions."""
    styles = set()
    for row in dataset:
        instr = row.get("instruction", "")
        m = re.search(r"style of '([^']+)'", instr)
        if m and _is_clean_style(m.group(1)):
            styles.add(m.group(1).strip())
            continue
        m = re.search(r"with cinematic style: '([^']+?)'", instr)
        if m and _is_clean_style(m.group(1)):
            styles.add(m.group(1).strip())
    return sorted(styles)


def build_seeds(styles: list[str]) -> list[dict]:
    """Build (style, axis, gen_idx) seed combinations."""
    seeds = []
    for style in styles:
        for axis in AXES:
            for gen_idx in range(GENERATIONS_PER_SEED):
                seeds.append({"style": style, "axis": axis, "gen_idx": gen_idx})
    return seeds


# =============================================================================
# STAGE 1: GENERATION
# =============================================================================

def make_user_prompt(seed: dict) -> str:
    return (
        f"STYLE NAME: '{seed['style']}'\n"
        f"AXIS: emphasize {seed['axis']}\n\n"
        "Write one image prompt now."
    )


GEN_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "response": {"type": "STRING"},
        "negative_prompt": {"type": "STRING"},
    },
    "required": ["response", "negative_prompt"],
}


def generate_one(client: genai.Client, seed: dict) -> dict | None:
    """Generate one prompt via Gemini. Returns None on failure."""
    try:
        resp = client.models.generate_content(
            model=GEN_MODEL,
            contents=make_user_prompt(seed),
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_GEN,
                temperature=0.85,
                top_p=0.9,
                max_output_tokens=4000,   # generous: thinking + JSON + 200-word prompt
                response_mime_type="application/json",
                response_schema=GEN_SCHEMA,
            ),
        )
        data = json.loads(resp.text)
        if not isinstance(data, dict) or "response" not in data:
            return None
        return {
            "style": seed["style"],
            "axis": seed["axis"],
            "response": data.get("response", "").strip(),
            "negative_prompt": data.get("negative_prompt", "").strip(),
        }
    except Exception as e:
        # Best-effort: skip failed generations rather than abort the run
        msg = str(e)[:200].replace("\n", " ")
        print(f"  ⚠ gen failed for style={seed['style'][:40]!r}: {type(e).__name__}: {msg}", file=sys.stderr)
        return None


def generate_all(client: genai.Client, seeds: list[dict]) -> list[dict]:
    """Generate all candidates in parallel."""
    print(f"🎨 Stage 1: generating {len(seeds)} candidates with {GEN_MODEL}")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(generate_one, client, s): s for s in seeds}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="generate"):
            r = fut.result()
            if r is not None:
                results.append(r)
    print(f"   {len(results)}/{len(seeds)} succeeded")
    return results


# =============================================================================
# STAGE 2: RULE-BASED FILTER
# =============================================================================

def passes_rules(text: str) -> bool:
    if not text or not text.strip():
        return False
    n_words = len(text.split())
    if not (MIN_WORDS <= n_words <= MAX_WORDS):
        return False
    for pat in BAD_PATTERNS:
        if re.search(pat, text, re.IGNORECASE | re.MULTILINE):
            return False
    return True


def rule_filter(records: list[dict]) -> list[dict]:
    print("🧹 Stage 2: rule-based filter")
    kept = [r for r in records if passes_rules(r["response"])]
    print(f"   {len(kept)}/{len(records)} passed")
    return kept


# =============================================================================
# STAGE 3: SEMANTIC DEDUPLICATION
# =============================================================================

def semantic_dedup(records: list[dict]) -> list[dict]:
    print(f"🔎 Stage 3: semantic dedup (cosine ≥ {DEDUP_THRESHOLD} → drop)")
    from sentence_transformers import SentenceTransformer

    if len(records) <= 1:
        return records
    encoder = SentenceTransformer(EMBED_MODEL)
    texts = [r["response"] for r in records]
    embeds = encoder.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    keep_mask = [True] * len(records)
    sim = cosine_similarity(embeds)
    for i in range(len(records)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(records)):
            if keep_mask[j] and sim[i, j] >= DEDUP_THRESHOLD:
                keep_mask[j] = False

    kept = [r for r, k in zip(records, keep_mask) if k]
    print(f"   {len(kept)}/{len(records)} unique")
    return kept


# =============================================================================
# STAGE 4: LLM-AS-JUDGE
# =============================================================================

JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "vocabulary_richness": {"type": "INTEGER"},
        "style_alignment": {"type": "INTEGER"},
        "compositional_clarity": {"type": "INTEGER"},
        "format_adherence": {"type": "INTEGER"},
    },
    "required": ["vocabulary_richness", "style_alignment", "compositional_clarity", "format_adherence"],
}


def judge_one(client: genai.Client, record: dict) -> dict | None:
    user_msg = (
        f"STYLE: '{record['style']}'\n"
        f"PROMPT: {record['response']}\n\n"
        "Rate now."
    )
    resp = None
    try:
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=user_msg,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_JUDGE,
                temperature=0.0,
                max_output_tokens=2000,
                response_mime_type="application/json",
                response_schema=JUDGE_SCHEMA,
                # Disable internal thinking — judge task is trivial (4 ints),
                # thinking tokens otherwise eat the entire budget before output.
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        scores = json.loads(resp.text)
        return {
            "vocabulary_richness": int(scores.get("vocabulary_richness", 0)),
            "style_alignment": int(scores.get("style_alignment", 0)),
            "compositional_clarity": int(scores.get("compositional_clarity", 0)),
            "format_adherence": int(scores.get("format_adherence", 0)),
        }
    except Exception as e:
        msg = str(e)[:200].replace("\n", " ")
        raw = (resp.text if resp is not None else "<no resp>")[:120]
        print(f"  ⚠ judge failed: {type(e).__name__}: {msg} | raw={raw!r}", file=sys.stderr)
        return None


def judge_all(client: genai.Client, records: list[dict]) -> list[dict]:
    print(f"⚖️  Stage 4: LLM-as-judge with {JUDGE_MODEL}")
    judged = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(judge_one, client, r): r for r in records}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="judge"):
            scores = fut.result()
            rec = futures[fut]
            if scores is None:
                continue
            avg = sum(scores.values()) / len(scores)
            min_dim = min(scores.values())
            if avg >= MIN_AVG_JUDGE_SCORE and min_dim >= MIN_DIM_JUDGE_SCORE:
                rec["judge_avg"] = avg
                rec["judge_scores"] = scores
                judged.append(rec)
    print(f"   {len(judged)}/{len(records)} passed (avg≥{MIN_AVG_JUDGE_SCORE} AND min_dim≥{MIN_DIM_JUDGE_SCORE})")
    return judged


# =============================================================================
# STAGE 5: FINALIZE + PUSH
# =============================================================================

def render_text(tokenizer, instruction: str, response: str, negative: str) -> str:
    """Render Qwen chat template with negative appended to assistant turn (matches v1 upload)."""
    assistant_content = f"{response}\n\nNegative: {negative}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TRAIN},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": assistant_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def finalize_records(records: list[dict], hf_token: str | None) -> list[dict]:
    """Build final v1-compatible schema with curated negatives fallback + Qwen-rendered text."""
    import random
    from transformers import AutoTokenizer

    rng = random.Random(SEED)
    print(f"📝 Stage 5a: rendering text field with {TRAIN_TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TRAIN_TOKENIZER_ID, token=hf_token)

    out = []
    for r in records:
        neg = r.get("negative_prompt") or rng.choice(CURATED_NEGATIVES)
        instruction = f"Generate a detailed image prompt in the style of '{r['style']}'."
        out.append({
            "instruction": instruction,
            "response": r["response"],
            "negative_prompt": neg,
            "text": render_text(tokenizer, instruction, r["response"], neg),
            "style": r["style"],
            "axis": r["axis"],
            "judge_avg": r["judge_avg"],
            "source": "synthetic_v2_gemini",
        })
    return out


def push_dataset(records: list[dict], hub_id: str, token: str) -> None:
    print(f"📤 Stage 5: pushing {len(records)} rows to {hub_id}")
    ds = Dataset.from_list(records)
    split = ds.train_test_split(test_size=0.10, seed=SEED)
    DatasetDict({"train": split["train"], "validation": split["test"]}).push_to_hub(
        hub_id, private=True, token=token
    )
    print(f"   🎉 https://huggingface.co/datasets/{hub_id}")


# =============================================================================
# ORCHESTRATION
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip Hub push; runs all stages on a small subset")
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="Limit number of seeds (default: all)")
    parser.add_argument("--source", default=V1_DATASET_ID)
    parser.add_argument("--target", default=V2_DATASET_ID)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_key:
        print("ERROR: set GEMINI_API_KEY (or GOOGLE_API_KEY)", file=sys.stderr)
        return 1
    if not hf_token and not args.dry_run:
        print("ERROR: HF_TOKEN required for Hub push (or use --dry-run)", file=sys.stderr)
        return 1

    print(f"🌱 Stage 0: loading source {args.source}")
    if os.path.isfile(args.source) and args.source.endswith((".json", ".jsonl")):
        src = load_dataset("json", data_files=args.source, split="train")
    else:
        try:
            src = load_dataset(args.source, split="train", token=hf_token)
        except Exception as e:
            if os.path.isfile(LOCAL_V1_JSONL):
                print(f"   ⚠ Hub load failed ({type(e).__name__}); falling back to {LOCAL_V1_JSONL}")
                src = load_dataset("json", data_files=LOCAL_V1_JSONL, split="train")
            else:
                raise
    styles = extract_styles(src)
    print(f"   {len(styles)} unique styles found")
    if not styles:
        print("ERROR: no styles extracted from source dataset", file=sys.stderr)
        return 1

    seeds = build_seeds(styles)
    if args.max_seeds:
        # Sample diverse seeds (different styles) for smoke tests instead of
        # taking the first N which would be all variants of the same style.
        import random
        rng = random.Random(SEED)
        rng.shuffle(seeds)
        seeds = seeds[:args.max_seeds]
    print(f"   {len(seeds)} seeds to generate (~{len(seeds)} candidates)")

    client = genai.Client(api_key=gemini_key)
    t0 = time.time()
    raw = generate_all(client, seeds)
    rule_passed = rule_filter(raw)
    deduped = semantic_dedup(rule_passed)
    judged = judge_all(client, deduped)
    final = finalize_records(judged, hf_token)

    print(f"\n📊 Pipeline summary (elapsed {time.time()-t0:.1f}s):")
    print(f"   seeds      → {len(seeds)}")
    print(f"   generated  → {len(raw)}")
    print(f"   rule-pass  → {len(rule_passed)}")
    print(f"   dedup      → {len(deduped)}")
    print(f"   judge-pass → {len(judged)}")
    print(f"   final      → {len(final)}")

    if final:
        print("\n🔎 Sample row:")
        sample = final[0]
        print(f"   instruction: {sample['instruction']}")
        print(f"   response   : {sample['response'][:200]}...")
        print(f"   negative   : {sample['negative_prompt'][:120]}")
        print(f"   judge_avg  : {sample['judge_avg']:.2f}")

    if args.dry_run:
        print("\n✅ Dry run complete (no Hub push).")
        return 0

    if not final:
        print("ERROR: no rows survived pipeline; aborting Hub push", file=sys.stderr)
        return 2

    push_dataset(final, args.target, hf_token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
