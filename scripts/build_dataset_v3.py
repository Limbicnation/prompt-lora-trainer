#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "duckdb>=1.0.0",
#   "datasets>=2.18.0",
#   "pandas>=2.0.0",
#   "tqdm>=4.65.0",
#   "thefuzz>=0.22.0",
#   "huggingface_hub>=0.22.0",
# ]
# ///
"""
Build Deforum Prompt Dataset v3

Cleans v2 dataset, augments with Creative Writing and Gutenberg Sci-Fi sources
via direct passage extraction (scoring paragraphs for visual/atmospheric density).

Usage:
    uv run scripts/build_dataset_v3.py --dry-run --seed 42
    uv run scripts/build_dataset_v3.py --target Limbicnation/deforum-prompt-lora-dataset-v3 --seed 42
"""

import argparse
import hashlib
import json
import os
import random
import re
from typing import Dict, List, Optional

import duckdb
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from thefuzz import fuzz
from tqdm import tqdm


# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with HF token if available."""
    conn = duckdb.connect()
    if HF_TOKEN:
        conn.execute("CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN ?);", [HF_TOKEN])
    return conn

# Keyword lists for visual/atmospheric scoring
SCORING_KEYWORDS = {
    "visual": [
        "light", "shadow", "glow", "shimmer", "dark", "bright", "haze", "fog",
        "smoke", "dust", "flame", "neon", "silhouette", "reflection", "beam",
        "radiance", "gleam", "flicker", "pulse", "luminescence"
    ],
    "atmospheric": [
        "silence", "whisper", "echo", "wind", "rain", "thunder", "creak",
        "hum", "pulse", "breathe", "stillness", "quiet", "hush", "murmur"
    ],
    "texture": [
        "grain", "rough", "smooth", "cold", "warm", "damp", "velvet", "rust",
        "glass", "metal", "concrete", "fabric", "tactile"
    ],
    "movement": [
        "drift", "float", "crawl", "sweep", "cascade", "ripple", "flicker",
        "sway", "surge", "flow", "glide", "linger"
    ],
    "scifi": [
        "hologram", "viewport", "console", "starfield", "nebula", "reactor",
        "dome", "corridor", "airlock", "hull", "cryo", "terminal", "interface"
    ]
}

# Replacement pools
SARAH_ALTERNATIVES = [
    "the figure", "the silhouette", "Elena", "Mara", "Yuki", "the woman",
    "the stranger", "the protagonist", "a shadow", "someone", "the subject",
    "a form", "the individual", "the character", "a person"
]

LIGHTING_ALTERNATIVES = [
    "rim lighting", "neon glow", "volumetric haze", "backlit silhouette",
    "split lighting", "candlelight", "diffused overhead", "practical lighting",
    "ambient bounce", "hard key light", "soft fill", "dramatic shadows",
    "low-key lighting", "atmospheric haze", "pools of light"
]

META_PATTERNS = [
    r"We respect the original creators.*?\.",
    r"Your descriptive prowess.*?\.",
    r"comes into view via",
    r"descriptive prowess",
    r"Scene\s+\d+\s*:",
]

# Tier configuration
TIER_WORD_COUNTS = {
    "short": (15, 30),
    "medium": (30, 60),
    "detailed": (60, 100),
}

# Qwen3 system message
SYSTEM_MESSAGE = "You are a cinematic video prompt generator specializing in the De Forum Art Film aesthetic."


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def score_text_for_visual_density(text: str, include_scifi: bool = False) -> float:
    """Score text for visual/atmospheric density."""
    text_lower = text.lower()
    score = 0
    
    for category, keywords in SCORING_KEYWORDS.items():
        if category == "scifi" and not include_scifi:
            continue
        for keyword in keywords:
            score += text_lower.count(keyword)
    
    # Normalize by word count
    word_count = count_words(text)
    if word_count > 0:
        score = score / (word_count ** 0.5)  # Square root to reward density without over-penalizing length
    
    return score


def diversify_sarah(text: str, rng: random.Random) -> str:
    """Replace 'Sarah' with alternatives from pool."""
    # Use word boundaries to avoid partial matches
    def replace_match(match):
        return rng.choice(SARAH_ALTERNATIVES)
    
    # Case-insensitive replacement preserving case patterns would be complex,
    # so we do simple replacement
    text = re.sub(r'\bSarah\b', lambda m: rng.choice(SARAH_ALTERNATIVES), text, flags=re.IGNORECASE)
    return text


def diversify_chiaroscuro(text: str, rng: random.Random, keep_ratio: float = 0.2) -> str:
    """Replace chiaroscuro with alternatives, keeping some percentage."""
    matches = list(re.finditer(r'\bchiaroscuro\b', text, flags=re.IGNORECASE))
    
    for match in reversed(matches):  # Reverse to preserve indices
        if rng.random() > keep_ratio:
            replacement = rng.choice(LIGHTING_ALTERNATIVES)
            start, end = match.span()
            text = text[:start] + replacement + text[end:]
    
    return text


def strip_meta_commentary(text: str) -> str:
    """Remove meta-commentary and boilerplate."""
    for pattern in META_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def format_chat_template_v2(
    instruction: str,
    response: str,
    negative: str = "",
    tags: str = "",
    aspect_ratio: str = "16:9",
    model: str = "WanVideo",
    seed: int = 42
) -> str:
    """Format with full technical parameters (for v2_cleaned source)."""
    text = f"""<|im_start|>system
{SYSTEM_MESSAGE}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}

Technical Parameters:
- Negative Prompt: {negative}
- Tags: {tags}
- Settings: --ar {aspect_ratio} --model {model} --seed {seed}<|im_end|>"""
    return text


def format_chat_template_simple(instruction: str, response: str) -> str:
    """Format without technical parameters (for supplementary sources)."""
    text = f"""<|im_start|>system
{SYSTEM_MESSAGE}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""
    return text


def assign_tier(word_count: int) -> str:
    """Assign tier based on word count."""
    if word_count < 30:
        return "short"
    elif word_count < 60:
        return "medium"
    else:
        return "detailed"


# =============================================================================
# STAGE 1: Clean v2 Dataset
# =============================================================================

def stage1_clean_v2(rng: random.Random, dry_run: bool = False) -> pd.DataFrame:
    """Clean and diversify the v2 dataset."""
    print("\n" + "="*60)
    print("STAGE 1: Cleaning v2 Dataset")
    print("="*60)
    
    # Load via DuckDB
    conn = get_duckdb_connection()
    
    path = "hf://datasets/Limbicnation/deforum-prompt-lora-dataset-v2@~parquet/default/train/*.parquet"
    print(f"Loading from: {path}")
    
    try:
        df = conn.execute("SELECT * FROM read_parquet(?)", [path]).fetchdf()
    except duckdb.Error as e:
        print(f"Error loading v2 dataset: {e}")
        conn.close()
        return pd.DataFrame()
    
    conn.close()
    
    print(f"Loaded {len(df)} rows")
    
    # Track original stats
    original_sarah_count = df['response'].str.contains('Sarah', case=False, na=False).sum()
    original_chiaroscuro_count = df['response'].str.contains('chiaroscuro', case=False, na=False).sum()
    
    # Clean each row
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning"):
        instruction = str(row.get('instruction', ''))
        response = str(row.get('response', ''))
        
        # Skip truncated rows
        if count_words(instruction) < 3 or count_words(response) < 3:
            continue
        
        # Diversify Sarah (both instruction and response)
        response = diversify_sarah(response, rng)
        instruction = diversify_sarah(instruction, rng)

        # Diversify chiaroscuro
        response = diversify_chiaroscuro(response, rng, keep_ratio=0.15)

        # Strip meta commentary
        response = strip_meta_commentary(response)
        instruction = strip_meta_commentary(instruction)
        
        # Get metadata
        negative = str(row.get('negative_prompt', ''))
        tags = str(row.get('tags', ''))
        if isinstance(row.get('tags'), list):
            tags = ', '.join(row.get('tags'))
        
        aspect_ratio = str(row.get('aspect_ratio', '16:9'))
        model = str(row.get('model', 'WanVideo'))
        seed = int(row.get('seed', 42)) if pd.notna(row.get('seed')) else 42
        
        # Format chat template
        text = format_chat_template_v2(instruction, response, negative, tags, aspect_ratio, model, seed)
        
        # Calculate word count and tier
        word_count = count_words(response)
        tier = assign_tier(word_count)
        
        rows.append({
            'instruction': instruction,
            'response': response,
            'tier': tier,
            'word_count': word_count,
            'text': text,
            'source': 'v2_cleaned',
            'negative_prompt': negative,
            'tags': tags,
            'camera_movement': str(row.get('camera_movement', '')),
        })
    
    result_df = pd.DataFrame(rows)
    
    # Stats
    new_sarah_count = result_df['response'].str.contains('Sarah', case=False, na=False).sum()
    new_chiaroscuro_count = result_df['response'].str.contains('chiaroscuro', case=False, na=False).sum()
    
    print(f"\nStage 1 Results:")
    print(f"  Original rows: {len(df)}")
    print(f"  Cleaned rows: {len(result_df)}")
    print(f"  Sarah mentions: {original_sarah_count} → {new_sarah_count} ({100*new_sarah_count/len(result_df) if len(result_df) > 0 else 0:.1f}%)")
    print(f"  Chiaroscuro mentions: {original_chiaroscuro_count} → {new_chiaroscuro_count} ({100*new_chiaroscuro_count/len(result_df) if len(result_df) > 0 else 0:.1f}%)")
    print(f"  Tier distribution: {result_df['tier'].value_counts().to_dict()}")
    
    return result_df


# =============================================================================
# STAGE 2: Creative Writing ShareGPT
# =============================================================================

def stage2_creative_writing(
    rng: random.Random,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Process Creative Writing ShareGPT dataset."""
    print("\n" + "="*60)
    print("STAGE 2: Creative Writing ShareGPT")
    print("="*60)
    
    # Load via DuckDB
    conn = get_duckdb_connection()
    
    path = "hf://datasets/ChaoticNeutrals/Creative_Writing-ShareGPT@~parquet/default/train/*.parquet"
    print(f"Loading from: {path}")
    
    try:
        df = conn.execute("SELECT * FROM read_parquet(?)", [path]).fetchdf()
    except duckdb.Error as e:
        print(f"Error loading Creative Writing dataset: {e}")
        conn.close()
        return pd.DataFrame()
    
    conn.close()
    
    print(f"Loaded {len(df)} conversations")
    
    # Extract first human+gpt exchange and score
    scored_items = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        conversations = row.get('conversations', [])
        
        # Handle different formats
        if isinstance(conversations, str):
            try:
                conversations = json.loads(conversations)
            except json.JSONDecodeError:
                continue
        
        if conversations is None or len(conversations) < 2:
            continue
        
        # Find first human+gpt pair
        human_msg = None
        gpt_msg = None
        
        for msg in conversations:
            if isinstance(msg, dict):
                from_field = msg.get('from', '')
                value = msg.get('value', '')
                
                if from_field in ['human', 'user'] and not human_msg:
                    human_msg = value
                elif from_field in ['gpt', 'assistant'] and not gpt_msg:
                    gpt_msg = value
                
                if human_msg and gpt_msg:
                    break
        
        if not human_msg or not gpt_msg:
            continue
        
        # Score GPT response
        score = score_text_for_visual_density(gpt_msg)
        
        scored_items.append({
            'human': human_msg,
            'gpt': gpt_msg,
            'score': score,
            'word_count': count_words(gpt_msg)
        })
    
    print(f"Extracted {len(scored_items)} valid conversations")
    
    # Sort by score and take top 20%
    scored_items.sort(key=lambda x: x['score'], reverse=True)
    top_count = max(int(len(scored_items) * 0.2), 100)
    top_items = scored_items[:top_count]
    
    print(f"Selected top {len(top_items)} by visual density score")
    
    # Extract cinematic passages directly (fast, no LLM needed).
    # The Creative Writing GPT responses already contain rich, atmospheric text.
    # We extract the best paragraph from each response and format it.
    rows = []
    for item in tqdm(top_items, desc="Extracting passages"):
        gpt_text = item['gpt']
        instruction = f"Generate a cinematic video prompt for: {item['human'][:100]}"

        # Split into paragraphs and score each for visual density
        paragraphs = [p.strip() for p in gpt_text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [gpt_text]

        best_para = ""
        best_score = -1
        for para in paragraphs:
            wc = count_words(para)
            if wc < 15 or wc > 120:
                continue
            score = score_text_for_visual_density(para)
            if score > best_score:
                best_score = score
                best_para = para

        # Fallback: use first paragraph truncated
        if not best_para:
            best_para = paragraphs[0] if paragraphs else gpt_text
            words = best_para.split()
            best_para = ' '.join(words[:80])

        # Trim to target length (40-80 words)
        words = best_para.split()
        if len(words) > 80:
            # Truncate at sentence boundary near 60-80 words
            truncated = ' '.join(words[:80])
            last_period = truncated.rfind('.')
            if last_period > len(' '.join(words[:40])):
                best_para = truncated[:last_period + 1]
            else:
                best_para = truncated

        response = best_para.strip()
        word_count = count_words(response)
        if word_count < 10:
            continue

        tier = assign_tier(word_count)
        text = format_chat_template_simple(instruction, response)

        rows.append({
            'instruction': instruction,
            'response': response,
            'tier': tier,
            'word_count': word_count,
            'text': text,
            'source': 'creative_writing',
        })

    result_df = pd.DataFrame(rows)
    print(f"Stage 2 Results: {len(result_df)} rows")
    if len(result_df) > 0:
        print(f"  Tier distribution: {result_df['tier'].value_counts().to_dict()}")
    
    return result_df


# =============================================================================
# STAGE 3: Gutenberg Sci-Fi
# =============================================================================

def stage3_gutenberg(
    rng: random.Random,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Process Gutenberg Sci-Fi dataset."""
    print("\n" + "="*60)
    print("STAGE 3: Gutenberg Sci-Fi")
    print("="*60)
    
    # Load via DuckDB
    conn = get_duckdb_connection()
    
    path = "hf://datasets/stevez80/Sci-Fi-Books-gutenberg@~parquet/default/train/*.parquet"
    print(f"Loading from: {path}")
    
    try:
        # Sample ~500 books
        df = conn.execute(
            "SELECT * FROM read_parquet(?) USING SAMPLE 500",
            [path]
        ).fetchdf()
    except duckdb.Error as e:
        print(f"Error loading Gutenberg dataset: {e}")
        conn.close()
        return pd.DataFrame()
    
    conn.close()
    
    print(f"Loaded {len(df)} books")
    
    # Process each book
    chunks = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        text = str(row.get('text', ''))
        title = str(row.get('title', 'Unknown'))
        
        # Normalize line endings (real and escaped) and strip Gutenberg headers/footers
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = text.replace('\\r\\n', '\n').replace('\\r', '\n')
        text = re.sub(r'\*\*\*\s*START OF.*?\*\*\*', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\*\*\*\s*END OF.*?\*\*\*', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Also strip Project Gutenberg license blocks
        text = re.sub(r'End of Project Gutenberg.*', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Produced by.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Split into paragraphs (handle various formats)
        # Try multiple paragraph separators
        for separator in ['\n\n\n', '\n\n', '\r\n\r\n']:
            if separator in text:
                paragraphs = text.split(separator)
                break
        else:
            # Fallback: split by single newlines if no double newlines
            paragraphs = text.split('\n')
        
        # Build chunks of 200-500 words by splitting the text
        words = text.split()
        i = 0
        while i < len(words):
            # Try to find a good chunk boundary (200-500 words)
            chunk_size = min(350, len(words) - i)  # Target ~350 words
            if chunk_size < 100:  # Skip very small remainders
                break
            
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if 200 <= len(chunk_words) <= 500:
                chunks.append({
                    'text': chunk_text,
                    'title': title,
                    'word_count': len(chunk_words)
                })
            
            i += chunk_size  # Move forward (can overlap if needed)
            
            # Safety limit: max 5 chunks per book
            if len([c for c in chunks if c['title'] == title]) >= 5:
                break
    
    print(f"Created {len(chunks)} chunks")
    
    # Score chunks
    scored_chunks = []
    for chunk in tqdm(chunks, desc="Scoring"):
        score = score_text_for_visual_density(chunk['text'], include_scifi=True)
        chunk['score'] = score
        scored_chunks.append(chunk)
    
    # Sort and take top ~700
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    top_chunks = scored_chunks[:700]
    
    print(f"Selected top {len(top_chunks)} chunks by visual density")
    
    # Extract cinematic passages directly (fast, no LLM needed).
    # Gutenberg sci-fi contains rich atmospheric descriptions we can extract.
    rows = []
    for chunk in tqdm(top_chunks, desc="Extracting passages"):
        chunk_text = chunk['text']

        # Build instruction from first 2 sentences
        sentences = chunk_text.split('.')[:2]
        hint = '. '.join(s.strip() for s in sentences if s.strip())
        instruction = f"Generate a cinematic video prompt inspired by: {hint[:150]}"

        # Split into paragraphs and score each for visual density
        paragraphs = [p.strip() for p in chunk_text.split('\n\n') if p.strip()]
        if not paragraphs:
            # Try single-newline split as fallback
            paragraphs = [p.strip() for p in chunk_text.split('\n') if p.strip()]
        if not paragraphs:
            paragraphs = [chunk_text]

        best_para = ""
        best_score = -1
        for para in paragraphs:
            wc = count_words(para)
            if wc < 15 or wc > 120:
                continue
            score = score_text_for_visual_density(para, include_scifi=True)
            if score > best_score:
                best_score = score
                best_para = para

        # Fallback: use first paragraph truncated
        if not best_para:
            best_para = paragraphs[0] if paragraphs else chunk_text
            words = best_para.split()
            best_para = ' '.join(words[:80])

        # Trim to target length (40-80 words) at sentence boundary
        words = best_para.split()
        if len(words) > 80:
            truncated = ' '.join(words[:80])
            last_period = truncated.rfind('.')
            if last_period > len(' '.join(words[:40])):
                best_para = truncated[:last_period + 1]
            else:
                best_para = truncated

        response = best_para.strip()
        word_count = count_words(response)
        if word_count < 10:
            continue

        tier = assign_tier(word_count)
        text = format_chat_template_simple(instruction, response)

        rows.append({
            'instruction': instruction,
            'response': response,
            'tier': tier,
            'word_count': word_count,
            'text': text,
            'source': 'gutenberg_scifi',
        })

    result_df = pd.DataFrame(rows)
    print(f"Stage 3 Results: {len(result_df)} rows")
    if len(result_df) > 0:
        print(f"  Tier distribution: {result_df['tier'].value_counts().to_dict()}")

    return result_df


# =============================================================================
# STAGE 4: Merge, Deduplicate, Upload
# =============================================================================

def compute_hash(text: str) -> str:
    """Compute hash for blocking in deduplication.
    Uses first 40 chars to reduce false block collisions on short texts.
    """
    return hashlib.md5(text[:40].lower().encode()).hexdigest()[:8]


def fuzzy_deduplicate(df: pd.DataFrame, threshold: int = 85) -> pd.DataFrame:
    """Fuzzy deduplicate responses within tier blocks."""
    print(f"\nDeduplicating {len(df)} rows...")
    
    # Add hash for blocking
    df = df.copy()
    df['hash'] = df['response'].apply(compute_hash)
    
    # Group by tier + hash
    keep_indices = []
    
    for (tier, hash_val), group in tqdm(df.groupby(['tier', 'hash']), desc="Deduplicating"):
        if len(group) <= 1:
            keep_indices.extend(group.index.tolist())
            continue
        
        # Compare within group
        responses = group['response'].tolist()
        indices = group.index.tolist()
        
        skip = set()
        for i in range(len(responses)):
            if i in skip:
                continue
            keep_indices.append(indices[i])
            
            # Find similar
            for j in range(i + 1, len(responses)):
                if j in skip:
                    continue
                
                score = fuzz.token_sort_ratio(responses[i], responses[j])
                if score >= threshold:
                    skip.add(j)
    
    result = df.loc[keep_indices].drop(columns=['hash'])
    print(f"Deduplicated: {len(df)} → {len(result)} rows ({len(df) - len(result)} removed)")
    
    return result


def validate_dataset(df: pd.DataFrame) -> Dict[str, any]:
    """Validate the final dataset."""
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    checks = {
        'total_rows': len(df),
        'has_instruction': (df['instruction'].str.len() > 0).all(),
        'has_response': (df['response'].str.len() > 0).all(),
        'has_chat_tokens': df['text'].str.contains('<\|im_start\|>', regex=True).all(),
        'word_counts_valid': True,
        'sarah_pct': 0.0,
        'chiaroscuro_pct': 0.0,
        'tier_dist': {},
        'source_dist': {},
    }
    
    # Check word counts
    for tier, (min_w, max_w) in TIER_WORD_COUNTS.items():
        tier_df = df[df['tier'] == tier]
        if len(tier_df) > 0:
            invalid = ((tier_df['word_count'] < min_w) | (tier_df['word_count'] > max_w)).sum()
            if invalid > len(tier_df) * 0.1:  # Allow 10% margin
                checks['word_counts_valid'] = False
                print(f"  ⚠️ {tier}: {invalid}/{len(tier_df)} rows outside {min_w}-{max_w} word range")
    
    # Check Sarah (rows with Sarah in instruction OR response)
    sarah_mask = (
        df['response'].str.contains('Sarah', case=False, na=False)
        | df['instruction'].str.contains('Sarah', case=False, na=False)
    )
    sarah_count = sarah_mask.sum()
    checks['sarah_pct'] = 100 * sarah_count / len(df) if len(df) > 0 else 0
    
    # Check chiaroscuro
    chiaroscuro_count = df['response'].str.contains('chiaroscuro', case=False, na=False).sum()
    checks['chiaroscuro_pct'] = 100 * chiaroscuro_count / len(df) if len(df) > 0 else 0
    
    # Tier distribution
    checks['tier_dist'] = df['tier'].value_counts().to_dict()
    
    # Source distribution
    checks['source_dist'] = df['source'].value_counts().to_dict()
    
    # Print results
    print(f"Total rows: {checks['total_rows']}")
    print(f"Has instruction: {'✅' if checks['has_instruction'] else '❌'}")
    print(f"Has response: {'✅' if checks['has_response'] else '❌'}")
    print(f"Has chat tokens: {'✅' if checks['has_chat_tokens'] else '❌'}")
    print(f"Word counts valid: {'✅' if checks['word_counts_valid'] else '❌'}")
    print(f"Sarah: {sarah_count} ({checks['sarah_pct']:.1f}%) {'✅' if checks['sarah_pct'] < 10 else '❌'}")
    print(f"Chiaroscuro: {chiaroscuro_count} ({checks['chiaroscuro_pct']:.1f}%) {'✅' if checks['chiaroscuro_pct'] < 15 else '❌'}")
    print(f"Tier distribution: {checks['tier_dist']}")
    print(f"Source distribution: {checks['source_dist']}")
    
    # Sample outputs
    print("\n" + "="*60)
    print("SAMPLE OUTPUTS")
    print("="*60)
    
    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        if len(source_df) > 0:
            sample = source_df.iloc[0]
            print(f"\n--- {source.upper()} ---")
            print(f"Instruction: {sample['instruction'][:100]}...")
            print(f"Response: {sample['response'][:150]}...")
            print(f"Word count: {sample['word_count']}, Tier: {sample['tier']}")
    
    return checks


def stage4_merge_and_push(
    dfs: List[pd.DataFrame],
    target: str,
    dry_run: bool = False
) -> Optional[DatasetDict]:
    """Merge all dataframes, deduplicate, and push to Hub."""
    print("\n" + "="*60)
    print("STAGE 4: Merge, Deduplicate, Upload")
    print("="*60)
    
    # Combine
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined: {len(combined)} rows")
    
    # Deduplicate (threshold 85 balances variety vs quality)
    deduped = fuzzy_deduplicate(combined, threshold=85)
    
    # Validate
    checks = validate_dataset(deduped)
    
    # Check targets
    all_pass = (
        checks['has_instruction'] and
        checks['has_response'] and
        checks['has_chat_tokens'] and
        checks['word_counts_valid'] and
        checks['sarah_pct'] < 10 and
        checks['chiaroscuro_pct'] < 15
    )
    
    if not all_pass:
        print("\n⚠️ Validation failed - not pushing to Hub")
        return None
    
    if dry_run:
        print("\n[DRY RUN] Would push to Hub:")
        print(f"  Dataset: {target}")
        print(f"  Rows: {len(deduped)}")
        return None
    
    # Create train/val split
    deduped = deduped.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(deduped) * 0.9)
    
    train_df = deduped.iloc[:split_idx]
    val_df = deduped.iloc[split_idx:]
    
    print(f"\nSplit: {len(train_df)} train, {len(val_df)} validation")
    
    # Create datasets
    train_ds = Dataset.from_pandas(train_df[['instruction', 'response', 'tier', 'word_count', 'text', 'source']])
    val_ds = Dataset.from_pandas(val_df[['instruction', 'response', 'tier', 'word_count', 'text', 'source']])
    
    dataset_dict = DatasetDict({
        'train': train_ds,
        'validation': val_ds
    })
    
    # Push to Hub
    print(f"\nPushing to HuggingFace Hub: {target}")
    try:
        dataset_dict.push_to_hub(target, token=HF_TOKEN, private=True)
        print(f"✅ Successfully pushed to: https://huggingface.co/datasets/{target}")
    except Exception as e:
        print(f"❌ Failed to push: {e}")
        return None
    
    return dataset_dict


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build Deforum Prompt Dataset v3")
    parser.add_argument('--dry-run', action='store_true', help='Run all stages, print stats, do not push')
    parser.add_argument('--skip-creative', action='store_true', help='Skip Stage 2 (Creative Writing)')
    parser.add_argument('--skip-gutenberg', action='store_true', help='Skip Stage 3 (Gutenberg)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--target', type=str, default='Limbicnation/deforum-prompt-lora-dataset-v3', help='Target dataset ID')
    args = parser.parse_args()
    
    print("="*60)
    print("BUILD DATASET v3")
    print("="*60)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'FULL'}")
    print(f"Seed: {args.seed}")
    print(f"Skip Creative: {args.skip_creative}")
    print(f"Skip Gutenberg: {args.skip_gutenberg}")
    print(f"Target: {args.target}")
    print("="*60)
    
    # Check HF token
    if not HF_TOKEN:
        print("⚠️ Warning: HF_TOKEN not set. Private datasets may not be accessible.")
    
    # Initialize RNG
    rng = random.Random(args.seed)
    
    # Run stages
    dataframes = []
    
    # Stage 1: Always run
    df1 = stage1_clean_v2(rng, dry_run=args.dry_run)
    if len(df1) > 0:
        dataframes.append(df1)
    
    # Stage 2: Creative Writing
    if not args.skip_creative:
        df2 = stage2_creative_writing(rng, dry_run=args.dry_run)
        if len(df2) > 0:
            dataframes.append(df2)
    
    # Stage 3: Gutenberg
    if not args.skip_gutenberg:
        df3 = stage3_gutenberg(rng, dry_run=args.dry_run)
        if len(df3) > 0:
            dataframes.append(df3)
    
    # Stage 4: Merge and push
    if dataframes:
        stage4_merge_and_push(dataframes, args.target, dry_run=args.dry_run)
    else:
        print("\n❌ No data to process!")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == "__main__":
    main()
