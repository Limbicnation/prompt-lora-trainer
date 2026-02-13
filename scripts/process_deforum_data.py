#!/usr/bin/env python3
"""
De Forum Data Processing Pipeline

Transforms raw narrative text from The Deforum Art Film project into a high-quality
instruction-following dataset for fine-tuning Qwen3-4B-Instruct-2507.

Usage:
    python scripts/process_deforum_data.py \
        --input_dir ./inputs \
        --output_file ./data/deforum_prompts_processed.json \
        --augmentation_factor 20 \
        --min_examples_per_scene 15
"""

import os
import re
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict


@dataclass
class Scene:
    """Represents a scene from the storyboard."""
    id: str
    title: str
    narrative: str
    frames: Dict[str, str]
    visual_elements: List[str]
    negative_prompts: str
    scene_type: str = "narrative"


@dataclass
class TrainingExample:
    """A single training example for the dataset."""
    instruction: str
    response: str
    style_name: str
    negative_prompt: str
    tags: List[str]
    camera_movement: str
    technical_params: Dict[str, Any]
    scene_context: str


# De Forum Aesthetic Vocabulary
CINEMATIC_TERMS = [
    "tracking shot", "dolly zoom", "Dutch angle", "low angle", "high angle",
    "over-the-shoulder", "wide shot", "close-up", "extreme close-up",
    "chiaroscuro lighting", "rim lighting", "silhouette", "backlit",
    "film grain", "anamorphic", "shallow depth of field", "bokeh",
    "slow motion", "time-lapse", "match cut", "whip pan"
]

DE_FORUM_TAGS = [
    "noir", "minimalist", "urban", "psychological", "cinematic", "art film",
    "surreal", "dystopian", "tech-noir", "emotional", "dramatic",
    "moody", "atmospheric", "contemplative", "mysterious", "dark",
    "neo-noir", "existential", "philosophical", "humanist"
]

CAMERA_MOVEMENTS = [
    "slow tracking shot following subject",
    "static wide shot with subtle zoom",
    "handheld camera with slight shake",
    "low angle dolly forward",
    "overhead crane shot descending",
    "360-degree orbit around subject",
    "slow push-in on face",
    "pan across landscape",
    "tilt up from ground level",
    "whip pan transition",
    "slow zoom out revealing context",
    "steadicam following movement"
]

ASPECT_RATIOS = ["16:9", "2.39:1", "1.85:1", "4:3"]
MODELS = ["WanVideo", "CogVideoX", "AnimateDiff", "Stable Video Diffusion"]

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, static, distorted, low resolution, modern elements, bright colors, "
    "muted, bad eyes, dull, hazy, muddy colors, mutated, deformed, noise, "
    "stock image, borders, frame, watermark, text, signature, username, "
    "cropped, out of frame, bad composition, poorly rendered face, "
    "poorly drawn hands, low resolution, cartoonish style, futuristic elements"
)


def extract_json_frames(text: str) -> Dict[str, str]:
    """Extract JSON frame descriptions from text, including multi-line blocks."""
    frames = {}
    # Match JSON-like blocks (potentially multi-line) using brace balancing
    brace_pattern = re.compile(r'\{[^{}]*\}', re.DOTALL)
    for match in brace_pattern.finditer(text):
        block = match.group(0)
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    if re.fullmatch(r'\d+', str(key)) and isinstance(value, str):
                        frames[str(key)] = value
        except (json.JSONDecodeError, ValueError):
            continue
    return frames


def extract_scene_headers(text: str) -> List[Tuple[str, str, str]]:
    """Extract scene headers and their content."""
    scenes = []
    
    # Clean up common noise patterns first
    text = re.sub(r'\{[^}]*\}', '', text)  # Remove JSON blocks from header extraction
    
    # Match scene headers like "Scene 1:", "Scene 1 :", "# Scene 1"
    pattern = r'(?:Scene\s+(\d+[A-Za-z]*)\s*:?|#\s*Scene\s+(\d+[A-Za-z]*))\s*\n?([^\n]*?)\n?((?:(?!Scene\s+\d+|#\s*Scene).)*?)(?=\n(?:Scene\s+\d+|#\s*Scene|$)|\Z)'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        scene_num = match[0] or match[1]
        title = match[2].strip() if match[2] else f"Scene {scene_num}"
        content = match[3].strip()
        # Clean up content
        content = re.sub(r'^\{\s*"', '', content)  # Remove leading JSON markers
        if len(content) > 50:  # Only include scenes with substantial content
            scenes.append((scene_num, title, content))
    
    return scenes


def extract_negative_prompts(text: str) -> str:
    """Extract negative prompt sections from text."""
    # Look for negative prompt sections
    patterns = [
        r'[Nn]egative(?:\s+[Pp]rompt)?s?:?\s*\n([^\n]+(?:\n[^\n]+)*)',
        r'--neg\s+([^\n]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return DEFAULT_NEGATIVE_PROMPT


def extract_visual_elements(text: str) -> List[str]:
    """Extract visual descriptors from text."""
    elements = []
    
    # Look for descriptive adjectives and nouns
    visual_patterns = [
        r'([a-z]+)\s+colors?',
        r'([a-z]+)\s+lighting',
        r'([a-z]+)\s+shadows?',
        r'([a-z]+)\s+textures?',
        r'(swirling|pulsing|glowing|flickering|dim|bright)\s+([a-z]+)',
    ]
    
    for pattern in visual_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                elements.append(' '.join(match).lower())
            else:
                elements.append(match.lower())
    
    return list(set(elements))


def normalize_timestamp(timestamp: str) -> str:
    """Convert millisecond timestamps to descriptive labels."""
    try:
        ms = int(timestamp)
        if ms < 100:
            return "opening"
        elif ms < 300:
            return "early scene"
        elif ms < 600:
            return "mid-scene"
        elif ms < 900:
            return "late scene"
        else:
            return "climax"
    except ValueError:
        return timestamp


def parse_input_files(input_dir: str) -> List[Scene]:
    """Parse all input files and extract scenes."""
    scenes = []
    input_path = Path(input_dir)
    
    text_files = list(input_path.glob("*.txt"))
    print(f"📁 Found {len(text_files)} input files")
    
    for file_path in text_files:
        print(f"   Processing: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract negative prompts (file-level)
        negative_prompt = extract_negative_prompts(content)
        
        # Extract JSON frames
        frames = extract_json_frames(content)
        
        # Extract scene headers
        scene_headers = extract_scene_headers(content)
        
        if scene_headers:
            for scene_num, title, scene_content in scene_headers:
                scene_id = f"scene_{scene_num}_{file_path.stem}"
                
                # Extract frames specific to this scene
                scene_frames = {}
                for ts, desc in frames.items():
                    if desc in scene_content or any(ts in line for line in scene_content.split('\n')):
                        scene_frames[ts] = desc
                
                # If no frames found, extract narrative text (skip JSON)
                if not scene_frames and scene_content:
                    # Clean content for narrative
                    narrative_text = re.sub(r'\{[^}]*\}', '', scene_content)
                    narrative_text = re.sub(r'\n+', ' ', narrative_text).strip()
                    if len(narrative_text) > 50:
                        scene_frames["0"] = narrative_text[:500]
                
                visual_elements = extract_visual_elements(scene_content)
                
                scenes.append(Scene(
                    id=scene_id,
                    title=title or f"Scene {scene_num}",
                    narrative=scene_content[:1000],
                    frames=scene_frames,
                    visual_elements=visual_elements,
                    negative_prompts=negative_prompt
                ))
        else:
            # No explicit scenes - treat entire file as one scene
            scene_id = f"scene_1_{file_path.stem}"
            
            # Try to split by major sections
            sections = re.split(r'\n\n\n+', content)
            
            for i, section in enumerate(sections[:5]):  # Limit to first 5 sections
                if len(section.strip()) > 100:
                    frames = extract_json_frames(section)
                    if not frames:
                        frames = {"0": section[:500]}
                    
                    visual_elements = extract_visual_elements(section)
                    
                    scenes.append(Scene(
                        id=f"{scene_id}_section_{i}",
                        title=f"{file_path.stem} - Section {i+1}",
                        narrative=section[:1000],
                        frames=frames,
                        visual_elements=visual_elements,
                        negative_prompts=negative_prompt
                    ))
    
    print(f"✅ Extracted {len(scenes)} scenes")
    return scenes


def deduplicate_scenes(scenes: List[Scene]) -> List[Scene]:
    """Remove duplicate scenes based on content similarity."""
    seen = set()
    unique_scenes = []
    
    for scene in scenes:
        # Create a fingerprint from title and first frame
        fingerprint = f"{scene.title}_{list(scene.frames.values())[0][:100] if scene.frames else ''}"
        
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique_scenes.append(scene)
    
    print(f"🔄 Deduplicated: {len(scenes)} → {len(unique_scenes)} scenes")
    return unique_scenes


def infer_camera_movement(scene: Scene) -> str:
    """Infer appropriate camera movement for a scene."""
    narrative_lower = scene.narrative.lower()
    
    # Check for movement cues in narrative
    if any(word in narrative_lower for word in ['follow', 'tracking', 'chase']):
        return "slow tracking shot following subject"
    elif any(word in narrative_lower for word in ['zoom', 'close', 'face']):
        return "slow push-in on face"
    elif any(word in narrative_lower for word in ['wide', 'landscape', 'city', 'view']):
        return "static wide shot with subtle zoom"
    elif any(word in narrative_lower for word in ['look up', 'ascend', 'rise']):
        return "tilt up from ground level"
    elif any(word in narrative_lower for word in ['run', 'escape', 'flee']):
        return "handheld camera with slight shake"
    else:
        return random.choice(CAMERA_MOVEMENTS)


def extract_tags(scene: Scene) -> List[str]:
    """Extract relevant tags for a scene."""
    tags = []
    narrative_lower = scene.narrative.lower()
    
    # Match against De Forum vocabulary
    for tag in DE_FORUM_TAGS:
        if tag in narrative_lower or tag in ' '.join(scene.visual_elements).lower():
            tags.append(tag)
    
    # Add scene-specific tags
    if 'night' in narrative_lower or 'dark' in narrative_lower:
        tags.append('night')
    if 'studio' in narrative_lower:
        tags.append('studio')
    if 'city' in narrative_lower or 'urban' in narrative_lower:
        tags.append('cityscape')
    if 'humanist' in narrative_lower or 'rebel' in narrative_lower:
        tags.append('rebellion')
    
    # Ensure at least some tags
    if len(tags) < 2:
        tags.extend(random.sample(DE_FORUM_TAGS, 3))
    
    return list(set(tags))[:5]  # Limit to 5 tags


def format_video_prompt(scene: Scene, camera: str, style: str = "cinematic_art_film") -> str:
    """Format a video diffusion prompt from scene data."""
    # Get the main frame description and clean it
    main_desc = list(scene.frames.values())[0] if scene.frames else scene.narrative[:300]
    # Clean up JSON artifacts and formatting
    main_desc = re.sub(r'"\d+":\s*"', '', main_desc)  # Remove JSON timestamp keys
    main_desc = re.sub(r'["{}]', '', main_desc)  # Remove JSON brackets and quotes
    main_desc = re.sub(r'\n+', ' ', main_desc).strip()[:300]
    
    # Clean scene title
    clean_title = re.sub(r'["{}]', '', scene.title).strip()
    if not clean_title or clean_title in ['{', '}', '']:
        clean_title = "De Forum Scene"
    
    # Extract key visual elements
    visual_desc = ', '.join(scene.visual_elements[:5]) if scene.visual_elements else ""
    
    # Build prompt with cinematic language
    prompt_parts = [
        f"Cinematic {style.replace('_', ' ')} style video",
        f"scene: {clean_title}",
    ]
    
    if visual_desc:
        prompt_parts.append(f"visuals: {visual_desc}")
    
    # Add narrative if substantial
    if len(main_desc) > 20:
        prompt_parts.append(f"description: {main_desc}")
    
    prompt_parts.extend([
        f"camera movement: {camera}",
        f"lighting: chiaroscuro, moody atmospheric lighting with dramatic shadows",
        f"film style: grain texture, anamorphic lens characteristics, shallow depth of field",
        f"mood: contemplative, mysterious, emotionally resonant, noir-influenced"
    ])
    
    return '. '.join(prompt_parts)


def create_instruction_response_pair(scene: Scene, variation_type: str = "standard") -> TrainingExample:
    """Create an instruction-response pair from a scene."""
    camera = infer_camera_movement(scene)
    tags = extract_tags(scene)
    
    # Different instruction variations
    if variation_type == "standard":
        instruction = f"""Generate a cinematic video prompt for:
Scene: {scene.title}
Context: {scene.narrative[:200]}
Camera Movement: {camera}
Style: De Forum Art Film aesthetic"""
    
    elif variation_type == "frame_focus":
        frame_desc = list(scene.frames.values())[0][:150] if scene.frames else scene.narrative[:150]
        instruction = f"""Create a detailed video diffusion prompt based on this frame description:
{frame_desc}

Style requirements: Noir-influenced, minimalist composition, art film aesthetic
Camera: {camera}"""
    
    elif variation_type == "style_transfer":
        instruction = f"""Transform this narrative into a cinematic video prompt:
{scene.narrative[:200]}

Apply De Forum aesthetic: moody lighting, psychological depth, urban atmosphere"""
    
    elif variation_type == "technical":
        instruction = f"""Generate a technical video prompt with parameters for:
Scene: {scene.title}
Visual style: {', '.join(tags[:3])}
Include: camera movement, lighting setup, and technical specifications"""
    
    else:
        instruction = f"""Create a De Forum style video prompt for scene '{scene.title}'
Context: {scene.narrative[:200]}"""
    
    # Generate response
    response = format_video_prompt(scene, camera)
    
    # Technical parameters
    technical_params = {
        "aspect_ratio": random.choice(ASPECT_RATIOS),
        "model": random.choice(MODELS),
        "seed": random.randint(1000, 999999),
        "guidance_scale": round(random.uniform(7.0, 8.5), 1),
        "steps": random.choice([25, 30, 50])
    }
    
    return TrainingExample(
        instruction=instruction.strip(),
        response=response,
        style_name=scene.title,
        negative_prompt=scene.negative_prompts or DEFAULT_NEGATIVE_PROMPT,
        tags=tags,
        camera_movement=camera,
        technical_params=technical_params,
        scene_context=scene.narrative[:500]
    )


def augment_scene(scene: Scene, augmentation_factor: int) -> List[TrainingExample]:
    """Create multiple training examples from a single scene with variations."""
    examples = []
    
    variation_types = ["standard", "frame_focus", "style_transfer", "technical"]
    
    for i in range(augmentation_factor):
        # Cycle through variation types
        variation = variation_types[i % len(variation_types)]
        
        # Create base example
        example = create_instruction_response_pair(scene, variation)
        
        # Apply augmentations
        if i > 0:
            # Modify camera movement
            if i % 3 == 0:
                example.camera_movement = random.choice(CAMERA_MOVEMENTS)
            
            # Modify tags slightly
            if i % 2 == 0:
                example.tags = list(set(example.tags + random.sample(DE_FORUM_TAGS, 2)))
            
            # Adjust technical params
            example.technical_params["seed"] = random.randint(1000, 999999)
            example.technical_params["guidance_scale"] = round(random.uniform(7.0, 8.5), 1)
        
        examples.append(example)
    
    return examples


def validate_example(example: TrainingExample) -> Tuple[bool, Dict[str, Any]]:
    """Validate a training example meets quality standards."""
    checks = {
        "instruction_length": 20 <= len(example.instruction.split()) <= 200,
        "response_length": 30 <= len(example.response.split()) <= 400,
        "has_negative_prompt": len(example.negative_prompt) > 0,
        "has_tags": len(example.tags) >= 2,
        "technical_params_valid": all(k in example.technical_params for k in ["aspect_ratio", "model", "seed", "guidance_scale"]),
        "no_modern_elements": "modern" not in example.response.lower() and "bright colors" not in example.response.lower(),
        "cinematic_language": any(term in example.response.lower() for term in ["cinematic", "film", "shot", "lighting", "camera"])
    }
    
    return all(checks.values()), checks


def generate_review_samples(examples: List[TrainingExample], num_samples: int = 50) -> List[Dict]:
    """Generate random samples for manual review."""
    samples = random.sample(examples, min(num_samples, len(examples)))
    return [asdict(ex) for ex in samples]


def save_dataset(examples: List[TrainingExample], output_file: str) -> None:
    """Save processed examples to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = [asdict(ex) for ex in examples]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved {len(examples)} examples to {output_file}")


def generate_report(examples: List[TrainingExample], output_dir: str) -> None:
    """Generate processing report with statistics."""
    report_path = Path(output_dir) / "processing_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics
    instruction_lengths = [len(ex.instruction.split()) for ex in examples]
    response_lengths = [len(ex.response.split()) for ex in examples]
    all_tags = [tag for ex in examples for tag in ex.tags]
    
    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    report = f"""De Forum Dataset Processing Report
==================================

Total Examples: {len(examples)}

Length Statistics:
- Instruction: avg={sum(instruction_lengths)/len(instruction_lengths):.1f}, min={min(instruction_lengths)}, max={max(instruction_lengths)}
- Response: avg={sum(response_lengths)/len(response_lengths):.1f}, min={min(response_lengths)}, max={max(response_lengths)}

Top Tags:
"""
    
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        report += f"  {tag}: {count}\n"
    
    report += f"""
Camera Movements:
"""
    cameras = {}
    for ex in examples:
        cameras[ex.camera_movement] = cameras.get(ex.camera_movement, 0) + 1
    
    for cam, count in sorted(cameras.items(), key=lambda x: x[1], reverse=True)[:5]:
        report += f"  {cam}: {count}\n"
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📝 Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Process De Forum Art Film data")
    parser.add_argument("--input_dir", type=str, default="./inputs", help="Input directory with raw text files")
    parser.add_argument("--output_file", type=str, default="./data/deforum_prompts_processed.json", help="Output JSON file")
    parser.add_argument("--augmentation_factor", type=int, default=20, help="Number of variations per scene")
    parser.add_argument("--min_examples_per_scene", type=int, default=15, help="Minimum examples per scene")
    parser.add_argument("--review_samples", type=int, default=50, help="Number of review samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    
    print("🎬 De Forum Data Processing Pipeline")
    print("=" * 50)
    
    # Parse input files
    print("\n📖 Phase 1: Extracting scenes from input files...")
    scenes = parse_input_files(args.input_dir)
    
    if not scenes:
        print("❌ No scenes found in input files!")
        return
    
    # Deduplicate
    print("\n🧹 Phase 2: Deduplicating scenes...")
    scenes = deduplicate_scenes(scenes)
    
    # Create training examples with augmentation
    print(f"\n✨ Phase 3: Creating training examples (augmentation_factor={args.augmentation_factor})...")
    all_examples = []
    
    for scene in scenes:
        examples = augment_scene(scene, args.augmentation_factor)
        all_examples.extend(examples)
    
    print(f"   Generated {len(all_examples)} raw examples")
    
    # Validate examples
    print("\n✅ Phase 4: Validating examples...")
    valid_examples = []
    failed_validations = 0
    
    for example in all_examples:
        is_valid, checks = validate_example(example)
        if is_valid:
            valid_examples.append(example)
        else:
            failed_validations += 1
    
    print(f"   Valid: {len(valid_examples)}, Failed: {failed_validations}")
    
    # Ensure minimum examples per scene
    if len(valid_examples) < len(scenes) * args.min_examples_per_scene:
        print(f"   ⚠️ Warning: Fewer examples than target. Consider increasing augmentation.")
    
    # Generate review samples
    print(f"\n👁️ Phase 5: Generating {args.review_samples} review samples...")
    review_samples = generate_review_samples(valid_examples, args.review_samples)
    review_path = Path(args.output_file).parent / "review_samples.json"
    with open(review_path, 'w', encoding='utf-8') as f:
        json.dump(review_samples, f, indent=2, ensure_ascii=False)
    print(f"   Saved to {review_path}")
    
    # Save dataset
    print("\n💾 Phase 6: Saving dataset...")
    save_dataset(valid_examples, args.output_file)
    
    # Generate report
    print("\n📊 Phase 7: Generating report...")
    generate_report(valid_examples, Path(args.output_file).parent)
    
    print("\n🎉 Processing complete!")
    print(f"   Total examples: {len(valid_examples)}")
    print(f"   Output: {args.output_file}")


if __name__ == "__main__":
    main()
