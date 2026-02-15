#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "duckdb>=1.0.0",
#   "huggingface_hub>=0.20.0",
#   "datasets>=2.14.0",
#   "pandas>=2.0.0",
#   "tqdm>=4.65.0",
# ]
# ///
"""
Reformat Deforum Dataset v2

Transforms the rigid, verbose Limbicnation/deforum-prompt-lora-dataset into 
varied-length outputs with natural language prompts across three tiers:
- Short: 15-30 words, single sentence
- Medium: 30-60 words, flowing prose  
- Detailed: 60-100 words, rich cinematic description

Usage:
    uv run scripts/reformat_dataset_v2.py \
        --source "Limbicnation/deforum-prompt-lora-dataset" \
        --target "Limbicnation/deforum-prompt-lora-dataset-v2" \
        --seed 42
"""

import os
import re
import json
import random
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import duckdb
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from tqdm import tqdm


# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")

# Tier distribution
TIER_DISTRIBUTION = {
    "short": 0.33,
    "medium": 0.34,
    "detailed": 0.33,
}

# Target word counts per tier
TIER_WORD_COUNTS = {
    "short": (15, 30),
    "medium": (30, 60),
    "detailed": (60, 100),
}

# Instruction templates per tier
INSTRUCTION_TEMPLATES = {
    "short": [
        "One-line prompt for {scene}",
        "Brief cinematic prompt: {scene}",
        "Quick prompt for {scene}",
        "Concise video prompt: {scene}",
        "One-sentence description of {scene}",
        "Short prompt for {scene}",
        "Brief visual: {scene}",
    ],
    "medium": [
        "Generate a video prompt for {scene}",
        "Cinematic prompt for {scene}",
        "Create a prompt for {scene}",
        "Describe {scene} as a video scene",
        "Write a cinematic description of {scene}",
        "Video scene: {scene}",
        "Cinematic shot of {scene}",
    ],
    "detailed": [
        "Generate a detailed cinematic prompt with camera movement, lighting, and mood for {scene}",
        "Create a rich video description including camera work and atmosphere for {scene}",
        "Detailed cinematic scene with technical parameters for {scene}",
        "Comprehensive video prompt with film style elements for {scene}",
        "Elaborate cinematic description with lighting and camera for {scene}",
        "Full video scene description with mood and technical details for {scene}",
        "Detailed shot description with cinematic language for {scene}",
    ],
}

# Response templates per tier
RESPONSE_TEMPLATES = {
    "short": [
        "{camera} on {subject}, {lighting}.",
        "{subject}, {camera}, {mood}.",
        "{camera} revealing {subject} in {lighting}.",
        "{subject} bathed in {lighting}, {camera}.",
        "{camera} through {subject}, {mood} atmosphere.",
        "{subject} under {lighting}, {camera}.",
    ],
    "medium": [
        "{camera} on {subject}. {lighting_cap} casts dramatic shadows. {film_style} {mood}.",
        "{subject} comes into view via {camera}. {lighting_cap} creates {mood} tension. {film_style}.",
        "Through {camera}, we see {subject}. {lighting_cap} with {mood} undertones. {film_style}.",
        "{camera} captures {subject}. {lighting_cap} emphasizes the {mood} mood. {film_style}.",
        "{subject} emerges in {camera}. {lighting_cap} builds {mood} atmosphere. {film_style}.",
    ],
    "detailed": [
        "{scene_prefix} {camera} reveals {subject}. {lighting_cap} with dramatic shadows cutting across the frame. "
        "{film_style_detailed}. The scene carries a {mood} quality, {mood_extension}.",
        
        "{scene_prefix} Through {camera}, {subject} takes shape. {lighting_cap} creates pools of darkness and light. "
        "{film_style_detailed}. A {mood} atmosphere pervades, {mood_extension}.",
        
        "{scene_prefix} {camera} frames {subject}. {lighting_cap} sculpts the scene with cinematic precision. "
        "{film_style_detailed}. The {mood} mood deepens as {mood_extension}.",
        
        "{scene_prefix} {subject} unfolds via {camera}. {lighting_cap} with noir-influenced shadows. "
        "{film_style_detailed}. {mood_cap} emotions surface, {mood_extension}.",
    ],
}


@dataclass
class ParsedResponse:
    """Parsed components from a structured response."""
    scene: str
    description: str
    camera_movement: str
    lighting: str
    film_style: str
    mood: str


def parse_structured_response(response: str) -> ParsedResponse:
    """
    Parse the rigid structured response format to extract components.
    
    Expected format:
    Cinematic cinematic art film style video. scene: X. description: [text].
    camera movement: [type]. lighting: [description]. film style: [description]. mood: [description]
    """
    # Default values
    scene = ""
    description = ""
    camera_movement = ""
    lighting = ""
    film_style = ""
    mood = ""
    
    # Extract scene
    scene_match = re.search(r'scene:\s*([^\.]+)', response, re.IGNORECASE)
    if scene_match:
        scene = scene_match.group(1).strip()
    
    # Extract description
    desc_match = re.search(r'description:\s*([^\.]+(?:\.[^\.]+)?)', response, re.IGNORECASE)
    if desc_match:
        description = desc_match.group(1).strip()
    
    # Extract camera movement
    camera_match = re.search(r'camera movement:\s*([^\.]+)', response, re.IGNORECASE)
    if camera_match:
        camera_movement = camera_match.group(1).strip()
    else:
        # Fallback: look for camera movement in the text
        camera_keywords = ["tracking", "push-in", "handheld", "crane", "static", "dolly", 
                          "pan", "tilt", "zoom", "aerial", "steadicam", "whip"]
        for keyword in camera_keywords:
            if keyword in response.lower():
                # Extract phrase containing the keyword
                pattern = rf'[^.]*{keyword}[^.]*\.?'
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    camera_movement = match.group(0).strip()
                    break
    
    # Extract lighting
    lighting_match = re.search(r'lighting:\s*([^\.]+(?:\.[^\.]+)?)', response, re.IGNORECASE)
    if lighting_match:
        lighting = lighting_match.group(1).strip()
    else:
        # Fallback: look for lighting keywords
        if "chiaroscuro" in response.lower():
            lighting = "chiaroscuro, moody atmospheric lighting with dramatic shadows"
        elif "moody" in response.lower():
            lighting = "moody atmospheric lighting"
    
    # Extract film style
    film_match = re.search(r'film style:\s*([^\.]+(?:\.[^\.]+)?)', response, re.IGNORECASE)
    if film_match:
        film_style = film_match.group(1).strip()
    else:
        # Fallback
        if "grain" in response.lower() or "anamorphic" in response.lower():
            film_style = "grain texture, anamorphic lens characteristics, shallow depth of field"
    
    # Extract mood
    mood_match = re.search(r'mood:\s*([^\.]+)', response, re.IGNORECASE)
    if mood_match:
        mood = mood_match.group(1).strip()
    else:
        # Fallback: look for mood keywords
        mood_keywords = ["contemplative", "mysterious", "tense", "intimate", "noir", 
                        "dramatic", "melancholic", "suspenseful", "ethereal"]
        for keyword in mood_keywords:
            if keyword in response.lower():
                mood = keyword
                break
    
    return ParsedResponse(
        scene=scene,
        description=description,
        camera_movement=camera_movement,
        lighting=lighting,
        film_style=film_style,
        mood=mood
    )


def extract_subject(scene: str, description: str) -> str:
    """Extract a concise subject from scene and description."""
    # Try to get something meaningful from the scene
    if scene:
        # Clean up common prefixes
        subject = re.sub(r'^(INT\.?|EXT\.?)\s*[A-Z\s]+\s*-\s*(DAY|NIGHT|DUSK|DAWN)', '', scene, flags=re.IGNORECASE)
        subject = subject.strip(" -:.")
        if subject:
            return subject
    
    # Fallback to description
    if description:
        # Take first sentence or first 10 words
        first_sent = description.split('.')[0]
        words = first_sent.split()[:10]
        return ' '.join(words)
    
    return "the scene"


def simplify_camera(camera: str) -> str:
    """Simplify camera movement description for lower tiers."""
    if not camera:
        return "slow movement"
    
    # Extract just the camera type
    camera_types = [
        (r'\bslow push-in\b', "slow push-in"),
        (r'\bpush-in\b', "push-in"),
        (r'\bhandheld\b', "handheld camera"),
        (r'\btracking\b', "tracking shot"),
        (r'\bcrane\b', "crane shot"),
        (r'\bstatic\b', "static shot"),
        (r'\bdolly\b', "dolly movement"),
        (r'\bpan\b', "panning shot"),
        (r'\bzoom\b', "zoom"),
        (r'\baerial\b', "aerial view"),
        (r'\bwhip\b', "whip pan"),
    ]
    
    camera_lower = camera.lower()
    for pattern, simplified in camera_types:
        if re.search(pattern, camera_lower):
            return simplified
    
    # Return first few words
    return ' '.join(camera.split()[:3])


def simplify_lighting(lighting: str) -> str:
    """Simplify lighting description for lower tiers."""
    if not lighting:
        return "dramatic lighting"
    
    if "chiaroscuro" in lighting.lower():
        return "chiaroscuro lighting"
    
    if "moody" in lighting.lower():
        return "moody lighting"
    
    if "dramatic" in lighting.lower():
        return "dramatic shadows"
    
    # Return a shorter version
    words = lighting.split()
    if len(words) > 5:
        return ' '.join(words[:5])
    return lighting


def extract_mood_keywords(mood: str) -> Tuple[str, str]:
    """Extract mood keywords and create an extension phrase."""
    if not mood:
        return "mysterious", "tension builds in the quiet moments"
    
    # Common mood keywords
    mood_map = {
        "contemplative": ("contemplative", "the silence speaking volumes"),
        "mysterious": ("mysterious", "secrets lurking in the shadows"),
        "tense": ("tense", "uncertainty hanging in the air"),
        "intimate": ("intimate", "emotions raw and unguarded"),
        "noir": ("noir-influenced", "darkness enveloping the scene"),
        "dramatic": ("dramatic", "each moment charged with meaning"),
        "melancholic": ("melancholic", "longing echoing through the frame"),
        "suspenseful": ("suspenseful", "danger imminent yet unseen"),
        "ethereal": ("ethereal", "reality bending at the edges"),
        "emotionally resonant": ("emotionally resonant", "feelings surfacing unbidden"),
    }
    
    mood_lower = mood.lower()
    for key, (adj, extension) in mood_map.items():
        if key in mood_lower:
            return adj, extension
    
    # Extract first adjective
    words = mood.split(',')
    if words:
        first_mood = words[0].strip()
        return first_mood, f"the {first_mood} atmosphere deepening"
    
    return "atmospheric", "the scene unfolding with quiet intensity"


def generate_short_response(parsed: ParsedResponse) -> str:
    """Generate a short (15-30 words) response."""
    subject = extract_subject(parsed.scene, parsed.description)
    camera = simplify_camera(parsed.camera_movement)
    lighting = simplify_lighting(parsed.lighting)
    mood_keywords, _ = extract_mood_keywords(parsed.mood)
    
    template = random.choice(RESPONSE_TEMPLATES["short"])
    
    # Fill template
    response = template.format(
        camera=camera,
        subject=subject,
        lighting=lighting,
        mood=mood_keywords
    )
    
    # Capitalize first letter
    response = response[0].upper() + response[1:]
    
    return response


def generate_medium_response(parsed: ParsedResponse) -> str:
    """Generate a medium (30-60 words) response."""
    subject = extract_subject(parsed.scene, parsed.description)
    camera = simplify_camera(parsed.camera_movement)
    lighting = simplify_lighting(parsed.lighting)
    lighting_cap = lighting[0].upper() + lighting[1:] if lighting else "Dramatic lighting"
    
    # Simplify film style
    film_parts = parsed.film_style.split(',') if parsed.film_style else ["grain texture"]
    film_style = film_parts[0].strip() if film_parts else "grain texture"
    if "grain" in film_style.lower():
        film_style = "Grain texture adds cinematic depth"
    elif "anamorphic" in film_style.lower():
        film_style = "Anamorphic lens characteristics create distinctive flares"
    elif "shallow" in film_style.lower():
        film_style = "Shallow depth of field isolates the subject"
    else:
        film_style = f"{film_style[0].upper() + film_style[1:]} creates visual texture"
    
    mood_keywords, _ = extract_mood_keywords(parsed.mood)
    
    template = random.choice(RESPONSE_TEMPLATES["medium"])
    
    response = template.format(
        camera=camera,
        subject=subject,
        lighting=lighting,
        lighting_cap=lighting_cap,
        film_style=film_style,
        mood=mood_keywords
    )
    
    return response


def generate_detailed_response(parsed: ParsedResponse) -> str:
    """Generate a detailed (60-100 words) response."""
    subject = extract_subject(parsed.scene, parsed.description)
    camera = parsed.camera_movement if parsed.camera_movement else "a slow deliberate movement"
    
    # Scene prefix (optional)
    scene_prefix = ""
    if parsed.scene and re.match(r'^(INT|EXT)', parsed.scene, re.IGNORECASE):
        scene_prefix = parsed.scene.strip() + ". "
    
    # Lighting
    lighting = parsed.lighting if parsed.lighting else "chiaroscuro lighting"
    lighting_cap = lighting[0].upper() + lighting[1:]
    
    # Film style detailed
    film_style = parsed.film_style if parsed.film_style else "grain texture, shallow depth of field"
    film_style_detailed = f"{film_style[0].upper() + film_style[1:]} contributes to the cinematic language"
    
    # Mood
    mood_keywords, mood_extension = extract_mood_keywords(parsed.mood)
    mood_cap = mood_keywords[0].upper() + mood_keywords[1:]
    
    template = random.choice(RESPONSE_TEMPLATES["detailed"])
    
    response = template.format(
        scene_prefix=scene_prefix,
        camera=camera,
        subject=subject,
        lighting=lighting,
        lighting_cap=lighting_cap,
        film_style_detailed=film_style_detailed,
        mood=mood_keywords.lower(),
        mood_cap=mood_cap,
        mood_extension=mood_extension
    )
    
    return response


def generate_instruction(scene: str, tier: str) -> str:
    """Generate an instruction appropriate for the tier."""
    # Extract a simplified scene description
    scene_clean = scene.strip(" -:.")
    
    # Remove scene headings if present
    scene_clean = re.sub(r'^(INT\.?|EXT\.?)\s*[A-Z\s]+\s*-\s*(DAY|NIGHT|DUSK|DAWN)\s*-?\s*', '', scene_clean, flags=re.IGNORECASE)
    
    if not scene_clean:
        scene_clean = "a cinematic scene"
    
    template = random.choice(INSTRUCTION_TEMPLATES[tier])
    return template.format(scene=scene_clean)


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def assign_tier(index: int, total: int, seed: int = 42) -> str:
    """Assign a tier based on index to achieve target distribution."""
    random.seed(seed + index)
    roll = random.random()
    
    if roll < TIER_DISTRIBUTION["short"]:
        return "short"
    elif roll < TIER_DISTRIBUTION["short"] + TIER_DISTRIBUTION["medium"]:
        return "medium"
    else:
        return "detailed"


def reformat_row(row: Dict, tier: str) -> Dict:
    """Reformat a single dataset row to the specified tier."""
    # Parse the existing response
    parsed = parse_structured_response(row.get("response", ""))
    
    # Generate new response based on tier
    if tier == "short":
        new_response = generate_short_response(parsed)
    elif tier == "medium":
        new_response = generate_medium_response(parsed)
    else:
        new_response = generate_detailed_response(parsed)
    
    # Generate matching instruction
    new_instruction = generate_instruction(parsed.scene, tier)
    
    # Extract tags - handle both list and string formats
    tags = row.get("tags", [])
    if isinstance(tags, list):
        tags_str = ", ".join(tags)
    else:
        tags_str = str(tags) if tags else ""
    
    # Create the reformatted row with all fields needed by training script
    reformatted = {
        # Original metadata (for training script compatibility)
        "camera_movement": row.get("camera_movement", ""),
        "scene_context": row.get("scene_context", ""),
        "scene": parsed.scene,  # Extracted scene
        "tags": tags_str,
        "style_name": row.get("style_name", "Cinematic"),
        "source_file": row.get("source_file", ""),
        "negative_prompt": row.get("negative_prompt", ""),
        
        # New fields (primary training fields)
        "instruction": new_instruction,
        "response": new_response,
        "tier": tier,
        "word_count": count_words(new_response),
        
        # Keep original for reference
        "original_instruction": row.get("instruction", ""),
        "original_response": row.get("response", ""),
    }
    
    return reformatted


def load_source_dataset(dataset_id: str, token: Optional[str] = None) -> pd.DataFrame:
    """Load the source dataset using DuckDB."""
    conn = duckdb.connect()
    
    if token:
        conn.execute(f"CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN '{token}');")
    
    # Build the path
    path = f"hf://datasets/{dataset_id}@~parquet/default/train/*.parquet"
    
    print(f"Loading dataset from: {path}")
    
    # Query all data
    query = f"SELECT * FROM '{path}'"
    df = conn.execute(query).fetchdf()
    
    conn.close()
    
    print(f"Loaded {len(df)} rows")
    return df


def reformat_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Reformat the entire dataset with varied tiers."""
    print(f"Reformatting {len(df)} rows with seed {seed}...")
    
    reformatted_rows = []
    tier_counts = {"short": 0, "medium": 0, "detailed": 0}
    
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Reformatting")):
        # Assign tier
        tier = assign_tier(idx, len(df), seed)
        tier_counts[tier] += 1
        
        # Reformat row
        row_dict = row.to_dict()
        reformatted = reformat_row(row_dict, tier)
        reformatted_rows.append(reformatted)
    
    # Create new DataFrame
    result_df = pd.DataFrame(reformatted_rows)
    
    # Print distribution
    print("\nTier distribution:")
    total = len(result_df)
    for tier, count in tier_counts.items():
        pct = count / total * 100
        word_range = TIER_WORD_COUNTS[tier]
        print(f"  {tier}: {count} rows ({pct:.1f}%) - target {word_range[0]}-{word_range[1]} words")
    
    # Print word count statistics
    print("\nWord count statistics:")
    print(f"  Short tier: mean={result_df[result_df['tier']=='short']['word_count'].mean():.1f} words")
    print(f"  Medium tier: mean={result_df[result_df['tier']=='medium']['word_count'].mean():.1f} words")
    print(f"  Detailed tier: mean={result_df[result_df['tier']=='detailed']['word_count'].mean():.1f} words")
    print(f"  Overall: min={result_df['word_count'].min()}, max={result_df['word_count'].max()}, mean={result_df['word_count'].mean():.1f}")
    
    return result_df


def push_to_hub(df: pd.DataFrame, target_id: str, token: Optional[str] = None, private: bool = True):
    """Push the reformatted dataset to HuggingFace Hub."""
    print(f"\nPushing to HuggingFace Hub: {target_id}")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    dataset_dict = DatasetDict({"train": dataset})
    
    # Push to hub
    dataset_dict.push_to_hub(
        target_id,
        token=token,
        private=private
    )
    
    print(f"Successfully pushed to: https://huggingface.co/datasets/{target_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Reformat Deforum dataset with varied-length outputs"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="Limbicnation/deforum-prompt-lora-dataset",
        help="Source dataset ID"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Limbicnation/deforum-prompt-lora-dataset-v2",
        help="Target dataset ID"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make the target dataset private"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: save to local parquet file instead of pushing to hub"
    )
    
    args = parser.parse_args()
    
    # Check for HF token
    token = HF_TOKEN
    if not token:
        print("Warning: HF_TOKEN not set. Private datasets may not be accessible.")
    
    # Load source dataset
    df = load_source_dataset(args.source, token)
    
    # Reformat
    result_df = reformat_dataset(df, seed=args.seed)
    
    # Save or push
    if args.output:
        result_df.to_parquet(args.output)
        print(f"\nSaved to: {args.output}")
    else:
        push_to_hub(result_df, args.target, token, args.private)
    
    # Print sample outputs
    print("\n" + "="*80)
    print("SAMPLE OUTPUTS")
    print("="*80)
    
    for tier in ["short", "medium", "detailed"]:
        tier_df = result_df[result_df["tier"] == tier]
        if len(tier_df) > 0:
            sample = tier_df.iloc[0]
            print(f"\n--- {tier.upper()} (Tier) ---")
            print(f"Instruction: {sample['instruction']}")
            print(f"Response: {sample['response']}")
            print(f"Word count: {sample['word_count']}")


if __name__ == "__main__":
    main()
