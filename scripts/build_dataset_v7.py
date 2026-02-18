#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.18.0",
#   "pandas>=2.0.0",
#   "tqdm>=4.65.0",
#   "thefuzz>=0.22.0",
#   "huggingface_hub>=0.22.0",
#   "ollama>=0.4.0",
# ]
# ///
"""
Build Deforum Prompt Dataset v7 — Film Supplement

Strategy:
  - Loads v6 base dataset (1,483 clean rows) from HuggingFace
  - Synthesizes ~200 fresh rows from De Forum Art Film narrative seeds
  - SARAH character name is allowed (intentional film content, confirmed by director)
  - All other v6 quality gates apply unchanged
  - Merges, deduplicates, 90/10 splits, pushes as v7

Why NOT importing from v2:
  v2's "SARAH scenes" are 2,887 rows all duplicated from one scene (Scene 12B)
  with corrupted instructions ("cene 12B //"). Fresh synthesis is cleaner.

Film narrative (De Forum Art Film):
  Sarah is a young artist. De Forum is a powerful corporate figure (also the
  name of an art space). Sarah confronts De Forum after her sister is harmed
  by his schemes. Urban noir aesthetic, chiaroscuro lighting.

Usage:
    # Dry-run (5 film seeds only, no push)
    conda run -n prompt-lora-trainer uv run scripts/build_dataset_v7.py --dry-run

    # Full run
    conda run -n prompt-lora-trainer uv run scripts/build_dataset_v7.py

    # Different Ollama model
    conda run -n prompt-lora-trainer uv run scripts/build_dataset_v7.py \\
        --ollama-model qwen3:8b
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional

import ollama as ollama_lib
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from thefuzz import fuzz
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")
V6_DATASET_ID = "Limbicnation/deforum-prompt-lora-dataset-v6"
V7_DATASET_ID = "Limbicnation/deforum-prompt-lora-dataset-v7"

SYSTEM_MESSAGE = (
    "You are a cinematic video prompt generator specializing in the De Forum Art Film aesthetic."
)

OLLAMA_MODEL = "qwen3:4b"
OLLAMA_OPTIONS = {
    "temperature": 0.85,
    "top_p": 0.9,
    "num_predict": 3000,
    "repeat_penalty": 1.3,
}

FILM_SYNTHESIS_PROMPT = """Write a cinematic video diffusion prompt for this De Forum Art Film scene.

{scene_seed}

40-70 words. Begin with camera movement. Film grain. Chiaroscuro lighting. \
Character names (e.g. Sarah) are allowed when integral to the scene. \
No technical specs. No screenplay format. No introductory phrases. Output ONLY the prompt."""


# =============================================================================
# DE FORUM ART FILM — SCENE SEEDS (~20 narrative beats)
# =============================================================================

DE_FORUM_FILM_SEEDS = [
    # --- Sarah: Studio ---
    "Sarah at her studio workstation, paintings glowing on multiple screens in the early hours",
    "Sarah asleep at her desk, screens still alive with her unfinished work, grey dawn creeping in",
    "Sarah's studio in disarray — evidence photographs and string on a corkboard, one lamp burning",
    "Sarah debugging code on three monitors, apartment dark around her, focused and alone",
    "Sarah reviewing footage on her laptop, face lit blue-white in the dark studio",
    # --- Sarah: The Confrontation ---
    "Sarah in the glass tower elevator, city rising below through the floor, rehearsing what to say",
    "Sarah's confrontation with De Forum in the boardroom, two silhouettes against floor-to-ceiling windows",
    "De Forum across the long table, city skyline behind him, Sarah's reflection caught in the glass",
    "the boardroom aftermath — overturned chairs, scattered documents, Sarah standing in the silence",
    "Sarah leaving the De Forum building for the last time, early morning light on the empty plaza",
    # --- Sarah: The Sister ---
    "Sarah's sister in the hospital room, venetian blind shadows crossing white sheets",
    "Sarah and her sister reunited at the hospital window, city behind them in soft focus",
    "Sarah walking the hospital corridor alone at night, strip lights and closed doors",
    # --- De Forum: The Building ---
    "De Forum Art Space exterior at dawn, Sarah approaching alone across the empty plaza",
    "De Forum Art Space exhibition opening — crowds in the atrium, Sarah watching from the edge",
    "De Forum building at night, a single lit window high up, everything else dark",
    "establishing shot of De Forum Art Space, Sarah's first approach — intimidating modernist facade",
    # --- De Forum: The Corporation ---
    "De Forum's executive corridor at midnight, Sarah moving past security camera blind spots",
    "De Forum's empty office after the fall — desk lamp still burning, papers across the floor",
    "protest outside De Forum tower, rain on upturned faces, neon signs blurring in wet glass",
    # --- Pursuit & Tension ---
    "Sarah running through the underground parking structure beneath De Forum tower",
    "rooftop handoff above the city — Sarah and an ally in fog, distant buildings disappearing below",
    "Sarah watching De Forum's building from a cafe across the street, rain on the window",
    "De Forum's security footage playback — grainy surveillance, Sarah's silhouette in corridors",
]


# =============================================================================
# INSTRUCTION TEMPLATES (film-aware variants of v6 templates)
# =============================================================================

FILM_INSTRUCTION_TEMPLATES = [
    "Generate a cinematic video prompt for: {scene}",
    "Write a De Forum art film prompt for: {scene}",
    "Cinematic video prompt: {scene}",
    "Video diffusion prompt for {scene}",
    "Create a cinematic prompt for this scene: {scene}",
    "De Forum aesthetic prompt for: {scene}",
    "Atmospheric video prompt: {scene}",
    "{scene}",
    "Create a video of: {scene}",
    "Visualize this scene: {scene}",
    "{scene} — describe this cinematically",
    "Video concept: {scene}",
]


# =============================================================================
# QUALITY GATES (v6 gates minus the character name filter for SARAH)
# =============================================================================

SD_PARAM_PATTERNS = [
    r"--ar\s+\d+", r"--model\s+\w", r"--seed\s+\d+",
    r"\b4K\b", r"\b8K\b", r"\b1080p\b", r"\b720p\b",
    r"\b\d+fps\b", r"\bfps\b", r"\bHDR\b",
    r"\bmotion blur\b", r"\bshallow depth of field\b", r"\bdynamic range\b",
]

SCREENPLAY_PATTERNS = [
    r"(?i)^Scene\s+\d+\s*:", r"(?i)^Aspect Ratio\s*:",
    r"(?i)^Camera\s*:\s+", r"(?i)^Movement\s*:\s+",
    r"(?i)^Lighting\s*:\s+", r"(?i)^Film Style\s*:", r"(?i)^Mood\s*:\s+",
]

SYNTHESIS_LEAKAGE_PATTERNS = [
    r"(?i)include the following elements?",
    r"(?i)requirements?\s*:",
    r"(?m)^[\d]+[.):\s]\s+\w",   # numbered lists: "1. ", "1) ", "1: ", "1 "
    r"(?m)^[a-z][.)]\s+\w",
    r"(?i)^here'?s",
    r"(?i)^certainly",
    r"(?i)^sure[.,!\s]",
    r"(?i)^i'?d be happy",
    r"(?i)What is the .{5,40} of (this|the)",  # Q&A loop (v6 bug fix)
    r"(?i)Describe this scene",
    r"(?i)Video aesthetic\s*:",                 # structured label prefix (v6 bug fix)
    r"(?i)Cinematic video prompt\s*:",          # self-referencing label (v6 bug fix)
]

# Allowed character: SARAH (intentional film content)
# Other generic names still rejected
OTHER_CHARACTER_NAMES = r"\b(?:John|Elena|Mara|Yuki|David|James|Michael|Emma|Alice|Bob|Maria|Anna)\b"


def count_words(text: str) -> int:
    return len(text.split())


def assign_tier(word_count: int) -> str:
    if word_count < 40:
        return "short"
    elif word_count < 70:
        return "medium"
    return "detailed"


def format_chat_template(instruction: str, response: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )


def apply_quality_gate(response: str, instruction: str) -> Optional[str]:
    """Return rejection reason or None if response passes."""
    wc = count_words(response)
    if wc < 30:
        return f"too_short:{wc}"
    if wc > 90:
        return f"too_long:{wc}"

    for p in SD_PARAM_PATTERNS:
        if re.search(p, response):
            return f"sd_param:{p}"

    for p in SCREENPLAY_PATTERNS:
        if re.search(p, response, re.MULTILINE):
            return f"screenplay:{p}"

    for p in SYNTHESIS_LEAKAGE_PATTERNS:
        if re.search(p, response, re.MULTILINE):
            return f"leakage:{p}"

    # Input echo
    resp_words = set(response.lower().split())
    inst_words = set(instruction.lower().split())
    if resp_words and len(resp_words & inst_words) / len(resp_words) > 0.5:
        return "input_echo"

    # Non-SARAH character names only
    if re.search(OTHER_CHARACTER_NAMES, response, re.IGNORECASE):
        return "other_character_name"

    # Repeated 5-gram
    words = response.lower().split()
    if len(words) >= 5:
        phrases: Dict[str, int] = {}
        for i in range(len(words) - 4):
            phrase = " ".join(words[i: i + 5])
            phrases[phrase] = phrases.get(phrase, 0) + 1
            if phrases[phrase] > 2:
                return "repeated_phrase"

    return None


def clean_response(text: str) -> str:
    """Strip thinking tokens, quotes, word-count annotations."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"^\s*\**\s*(?:Prompt|Output)\s*\d*\s*:?\s*\**\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[\(\[—–-]\s*\d+\s*words?\s*[\)\]]?\s*$", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def ollama_generate(scene_seed: str, model: str, debug: bool = False) -> Optional[str]:
    prompt = FILM_SYNTHESIS_PROMPT.format(scene_seed=scene_seed)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(3):
        try:
            resp = ollama_lib.chat(model=model, messages=messages, options=OLLAMA_OPTIONS)
            content = (resp.message.content or "").strip()
            cleaned = clean_response(content)
            if debug:
                print(f"    [DEBUG] {count_words(cleaned)}w: {cleaned[:150]}")
            if count_words(cleaned) < 15:
                if attempt < 2:
                    time.sleep(1)
                    continue
                return None
            if count_words(cleaned) > 120:
                cleaned = " ".join(cleaned.split()[:90]) + "."
            return cleaned
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            print(f"  Ollama error: {e}")
            return None
    return None


def compute_hash(text: str) -> str:
    return hashlib.md5(text[:40].lower().encode()).hexdigest()[:8]


def fuzzy_deduplicate(df: pd.DataFrame, existing_responses: set, threshold: int = 85) -> pd.DataFrame:
    """Remove duplicates within new rows and against existing v6 responses."""
    print(f"\nDeduplicating {len(df)} new rows against {len(existing_responses)} v6 responses...")
    df = df.copy()

    keep = []
    seen_texts = set()

    for idx, row in df.iterrows():
        resp = row["response"]
        resp_lower = resp.lower()

        # Exact dedup against v6
        is_dup = False
        for ex in existing_responses:
            if fuzz.token_sort_ratio(resp_lower, ex.lower()) >= threshold:
                is_dup = True
                break

        if is_dup:
            continue

        # Dedup within new rows
        for seen in seen_texts:
            if fuzz.token_sort_ratio(resp_lower, seen) >= threshold:
                is_dup = True
                break

        if not is_dup:
            keep.append(idx)
            seen_texts.add(resp_lower)

    result = df.loc[keep].reset_index(drop=True)
    print(f"  {len(df)} → {len(result)} ({len(df) - len(result)} duplicates removed)")
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build Deforum Prompt Dataset v7 (v6 base + De Forum film scenes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Synthesize 5 film seeds only, no push")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", type=str, default=V7_DATASET_ID)
    parser.add_argument("--v6-source", type=str, default=V6_DATASET_ID)
    parser.add_argument("--film-limit", type=int, default=None,
                        help="Max film rows to synthesize (default: all seeds × 3 variants)")
    parser.add_argument("--ollama-model", type=str, default=OLLAMA_MODEL)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("BUILD DATASET v7 (v6 base + De Forum Film Scenes)")
    print("=" * 60)
    print(f"Mode:       {'DRY RUN' if args.dry_run else 'FULL'}")
    print(f"Seed:       {args.seed}")
    print(f"Model:      {args.ollama_model}")
    print(f"V6 source:  {args.v6_source}")
    print(f"Target:     {args.target}")
    print(f"Film seeds: {len(DE_FORUM_FILM_SEEDS)}")

    # Check Ollama
    print("\nChecking Ollama...")
    try:
        available = [m.model for m in ollama_lib.list().models]
        if not any(args.ollama_model in m for m in available):
            print(f"  Model '{args.ollama_model}' not found. Available: {available}")
            sys.exit(1)
        print(f"  Model '{args.ollama_model}' available")
    except Exception as e:
        print(f"  Cannot connect to Ollama: {e}")
        sys.exit(1)

    # Load v6 base
    print(f"\nLoading v6 base from {args.v6_source}...")
    try:
        v6 = load_dataset(args.v6_source, token=HF_TOKEN)
        v6_train = v6["train"].to_pandas()
        v6_val = v6.get("validation", v6["train"].select([])).to_pandas()
        print(f"  Loaded: {len(v6_train)} train, {len(v6_val)} validation rows")
    except Exception as e:
        print(f"  Failed to load v6: {e}")
        sys.exit(1)

    existing_responses = set(v6_train["response"].str.lower().tolist())

    # Synthesize film scenes
    print("\n" + "=" * 60)
    print("FILM SCENE SYNTHESIS (SARAH allowed)")
    print("=" * 60)

    rng = random.Random(args.seed)
    seeds = list(DE_FORUM_FILM_SEEDS)

    if args.dry_run:
        seeds = seeds[:5]
        print(f"  [DRY RUN] {len(seeds)} seeds")

    # Each seed gets 3 instruction template variants for diversity
    tasks = []
    for seed in seeds:
        templates = rng.sample(FILM_INSTRUCTION_TEMPLATES, min(3, len(FILM_INSTRUCTION_TEMPLATES)))
        for tmpl in templates:
            scene_truncated = seed[:150].rstrip(",. ")
            instruction = tmpl.format(scene=scene_truncated)
            tasks.append((seed, instruction))

    if args.film_limit:
        tasks = tasks[: args.film_limit]

    print(f"Synthesizing {len(tasks)} film rows ({len(seeds)} seeds × ~3 variants)...")
    rows = []
    rejected = 0

    for scene_seed, instruction in tqdm(tasks, desc="Film synthesis"):
        response = ollama_generate(scene_seed, model=args.ollama_model, debug=args.debug)
        if response is None:
            rejected += 1
            continue

        reason = apply_quality_gate(response, instruction)
        if reason:
            if args.debug:
                print(f"  GATE ({reason}): {response[:80]}")
            rejected += 1
            continue

        wc = count_words(response)
        rows.append({
            "instruction": instruction,
            "response": response,
            "tier": assign_tier(wc),
            "word_count": wc,
            "text": format_chat_template(instruction, response),
            "source": "de_forum_film",
        })

    print(f"\nFilm synthesis: {len(rows)} kept, {rejected} rejected")

    if not rows:
        print("\nNo film rows synthesized. Check Ollama and retry.")
        if not args.dry_run:
            sys.exit(1)

    film_df = pd.DataFrame(rows)
    if len(film_df) > 0:
        film_df = fuzzy_deduplicate(film_df, existing_responses)
        print(f"Tier distribution: {film_df['tier'].value_counts().to_dict()}")
        print(f"Word count: mean={film_df['word_count'].mean():.1f}")
        print("\n--- SAMPLE FILM OUTPUTS ---")
        for _, row in film_df.head(3).iterrows():
            print(f"\nInstruction: {row['instruction']}")
            print(f"Response:    {row['response']}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would add {len(film_df)} film rows to {len(v6_train)} v6 rows "
              f"and push to: {args.target}")
        return

    # Merge v6 + film rows
    combined_train = pd.concat([v6_train, film_df], ignore_index=True)
    combined_train = combined_train.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print(f"\nCombined train: {len(combined_train)} rows")
    print(f"Source distribution: {combined_train['source'].value_counts().to_dict()}")

    # Rebuild validation: keep v6 val + 10% of new film rows
    if len(film_df) > 10:
        film_val_idx = int(len(film_df) * 0.1)
        film_val = film_df.iloc[:film_val_idx]
        film_train = film_df.iloc[film_val_idx:]
        combined_val = pd.concat([v6_val, film_val], ignore_index=True)
        combined_train = pd.concat([v6_train, film_train], ignore_index=True)
        combined_train = combined_train.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    else:
        combined_val = v6_val.copy()

    print(f"Train: {len(combined_train)}, Validation: {len(combined_val)}")

    columns = ["instruction", "response", "tier", "word_count", "text", "source"]
    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(combined_train[columns]),
        "validation": Dataset.from_pandas(combined_val[columns]),
    })

    print(f"\nPushing to Hub: {args.target}")
    try:
        dataset_dict.push_to_hub(args.target, token=HF_TOKEN, private=True)
        print(f"Pushed: https://huggingface.co/datasets/{args.target}")
    except Exception as e:
        print(f"Push failed: {e}")
        local_path = f"./data/{args.target.split('/')[-1]}"
        os.makedirs(local_path, exist_ok=True)
        combined_train.to_json(f"{local_path}/train.jsonl", orient="records", lines=True)
        combined_val.to_json(f"{local_path}/validation.jsonl", orient="records", lines=True)
        print(f"Saved locally: {local_path}/")
        sys.exit(1)


if __name__ == "__main__":
    main()
