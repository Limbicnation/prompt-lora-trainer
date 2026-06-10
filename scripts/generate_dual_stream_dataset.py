#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "google-genai>=0.3.0",
#   "datasets>=2.18.0",
#   "huggingface-hub>=0.22.0",
#   "tqdm>=4.65.0",
#   "python-dotenv>=1.0.0",
#   "ollama>=0.4.0",
#   "transformers>=4.40.0",
#   "jinja2>=3.1.0",
# ]
# ///
"""Generate multi-dialect image-prompt SFT rows (concept + target_model → prompt).

Model routing happens at INFERENCE TIME via an instruction prefix, not in the output
structure. A single LLM call produces four prompt structures per concept; the script
fans them out into independent flat rows keyed by `target_model`, gating each dialect
separately (a failed dialect doesn't discard the others). This keeps the LoRA's task
simple — "given a concept and a target model, produce the right prompt dialect" — and
maximizes structural diversity for generalization.

Row schema:
    {
      "original_concept": "...",      # raw input / source style
      "target_model": "flux_t5",      # one of the 4 dialects below
      "instruction": "Generate a FLUX (T5-XXL) image prompt for: <concept>",
      "response": "..."               # the prompt in that dialect
    }

Two dialect sets, selected with --dialect-set (default image):
  image (Stable Diffusion / FLUX):
    flux_t5            dense natural-language prose, 40-120w
    sdxl_dual_clip     front-loaded comma tokens, 15-60w
    compact_caption    25-50w enriched caption (≥1 lighting term)
    steering_modifiers 5-8 comma-separated tags
  video (LTX-Video / WanVideo):
    wan_video          Subject+Scene+Motion prose, 20-80w (≥1 camera + ≥1 lighting term)
    ltx_video          motion-centric shot prose, 20-80w (≥1 camera term)
    compact_caption    ≤25w global caption
  Video dialects additionally HARD-REJECT non-visual tokens (scent/sound/memory/...),
  which contaminate 47.1% of the deforum-v7 source responses.

Two input modes (auto-detected):
  TRANSFORM (primary) — --dataset rows with a `response` column (e.g.
    Limbicnation/images-diffusion-prompt-style-v2). The vetted `response` becomes the
    flux_t5 dialect directly; the LLM only derives the other three. Cheaper and anchored
    to human-judge-validated content.
  GENERATE — --input / stdin rows with a `description` column. The LLM produces all four.

Calibration (v2 dataset, 2026-06-08): existing `response` is FLUX-T5-shaped (median 81w,
90.6% in 40-120w). Lighting term present 85.5% (hard gate ok); camera term only 53.5%
(SOFT/warn-only to avoid dropping ~47% of 4.8-star prompts). Buzzwords 1.5% (reject free).

Usage:
    # Smoke test: transform 10 v2 rows, validate only, no write/upload
    uv run scripts/generate_dual_stream_dataset.py \
        --dataset Limbicnation/images-diffusion-prompt-style-v2 \
        --max-rows 10 --dry-run

    # Transform full v2 → render text → push as new dataset
    uv run scripts/generate_dual_stream_dataset.py \
        --dataset Limbicnation/images-diffusion-prompt-style-v2 \
        --render-text --hub-id Limbicnation/dual-stream-image-prompts

    # Video dialects: transform the deforum v7 corpus (Ollama, dry-run smoke test)
    uv run scripts/generate_dual_stream_dataset.py \
        --dataset Limbicnation/deforum-prompt-lora-dataset-v7 \
        --dialect-set video --backend ollama --max-rows 20 --dry-run

    # Generate from raw descriptions
    uv run scripts/generate_dual_stream_dataset.py \
        --input data/raw_descriptions.jsonl --output data/dual_stream.jsonl

    # Single description via stdin
    echo '{"description": "a woman in a red dress standing in rain"}' \
        | uv run scripts/generate_dual_stream_dataset.py

    # Ollama backend
    uv run scripts/generate_dual_stream_dataset.py \
        --input data/raw_descriptions.jsonl --backend ollama --ollama-model qwen3:4b
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Banner is cosmetic — failures must never block generation.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from banner import print_banner

    print_banner()
except Exception:
    pass


# =============================================================================
# CONFIG
# =============================================================================

GEN_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
TRAIN_TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"  # for optional --render-text
MAX_CONCURRENT = 16
SEED = 42

DEFAULT_OUTPUT = str(
    Path(__file__).resolve().parent.parent / "data" / "dual_stream_records.jsonl"
)

SYSTEM_PROMPT_TRAIN = (
    "You are an expert image prompt generator for Stable Diffusion / FLUX. "
    "Given a concept and a target model, produce the prompt in that model's dialect. "
    "No labels, no preamble, no command-line flags."
)

# Dialect → (instruction prefix, bundle key returned by the LLM/assembly).
DIALECTS: dict[str, str] = {
    "flux_t5": "Generate a FLUX (T5-XXL) image prompt for:",
    "sdxl_dual_clip": "Generate an SDXL image prompt for:",
    "compact_caption": "Write a compact descriptive caption for:",
    "steering_modifiers": "List image steering modifiers for:",
}

# Gate thresholds (calibrated against v2 dataset — see module docstring)
CAPTION_MIN_W, CAPTION_MAX_W = 25, 50          # target 30-40
MODIFIER_MIN, MODIFIER_MAX = 5, 8
FLUX_MIN_W, FLUX_MAX_W = 40, 120
SDXL_MIN_W, SDXL_MAX_W = 15, 60

LIGHTING_TERMS = [
    "rim light", "key light", "backlight", "back light", "fill light", "ambient",
    "chiaroscuro", "volumetric", "rembrandt", "high-key", "high key", "low-key",
    "soft light", "hard light", "golden hour", "blue hour", "god rays",
    "subsurface scattering", "bioluminescent", "neon glow", "spotlight",
]
CAMERA_TERMS = [
    "35mm", "50mm", "85mm", "24mm", "anamorphic", "bokeh", "depth of field",
    "shallow focus", "focal length", "wide-angle", "wide angle", "macro",
    "telephoto", "fisheye", "barrel distortion", "tilt-shift", "long exposure",
    "f/1", "f/2", "f/4", "lens",
]
BUZZWORDS = ["stunning", "beautiful", "amazing", "gorgeous", "incredible", "breathtaking"]
GENERIC_MODIFIERS = {"art", "photo", "image", "picture", "render", "artwork", "drawing"}
SDXL_BOILERPLATE_HEADS = {
    "masterpiece", "best", "8k", "4k", "ultra", "highly", "high", "trending",
    "award", "professional", "hyperrealistic", "photorealistic", "detailed",
}

# Forbidden across every field (hard reject) — CLI flags, markdown, preamble.
BAD_PATTERNS = [
    r"--ar\s+\d", r"--seed\s+\d", r"--model\s+\w+", r"--steps\s+\d",
    r"--cfg\s+\d", r"--sampler\s+\w+",
    r"<lora:[^>]+>", r"<embedding:[^>]+>",
    r"```", r"^\s*#{1,6}\s", r"^\s*\*\*",
    r"^\s*(Here is|Here's|Sure,|Certainly|Of course|I'll|I will)",
]


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

_SCHEMA_DESC = """Output STRICT JSON. No markdown fences, no preamble.

Field rules:
- niche_compact_caption: 30-40 words (hard 25-50). Objective, specific. State spatial
  relationships, exact textures, a concrete lighting vector, and a camera/lens detail.
  FORBIDDEN filler: "stunning", "beautiful", "amazing", "gorgeous", "incredible".
- latent_steering_modifiers: 5-8 comma-separated, lowercase, hyper-specific tags.
  No generic words (art, photo, image, picture, render). No duplicates.
- flux_t5: dense NATURAL-LANGUAGE prose, 40-120 words, narrative flow (FLUX T5-XXL style).
  No bullet points, no numbered lists.
- sdxl_dual_clip: 15-60 words, comma-separated tokens FRONT-LOADED with the key subject
  and scene first, aesthetic/quality tokens last (SDXL dual-clip style)."""

SYSTEM_GENERATE = (
    "You are an expert prompt engineer for diffusion image models (FLUX, SDXL). "
    "Given a raw image description, produce four prompt structures for it.\n\n"
    + _SCHEMA_DESC
    + "\n\nReturn JSON with keys: niche_compact_caption, latent_steering_modifiers, "
    'model_routing_prompts (an object with keys flux_t5 and sdxl_dual_clip).'
)

SYSTEM_TRANSFORM = (
    "You are an expert prompt engineer for diffusion image models (FLUX, SDXL). "
    "You are given an existing, high-quality FLUX-style image prompt. Derive the other "
    "prompt structures from it WITHOUT changing its subject matter.\n\n"
    + _SCHEMA_DESC
    + "\n\nReturn JSON with keys: niche_compact_caption, latent_steering_modifiers, "
    "sdxl_dual_clip. (flux_t5 is taken from the source prompt, do not return it.)"
)


# =============================================================================
# GEMINI SCHEMAS
# =============================================================================

_ROUTING_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "flux_t5": {"type": "STRING"},
        "sdxl_dual_clip": {"type": "STRING"},
    },
    "required": ["flux_t5", "sdxl_dual_clip"],
}

GENERATE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "niche_compact_caption": {"type": "STRING"},
        "latent_steering_modifiers": {"type": "STRING"},
        "model_routing_prompts": _ROUTING_SCHEMA,
    },
    "required": [
        "niche_compact_caption",
        "latent_steering_modifiers",
        "model_routing_prompts",
    ],
}

TRANSFORM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "niche_compact_caption": {"type": "STRING"},
        "latent_steering_modifiers": {"type": "STRING"},
        "sdxl_dual_clip": {"type": "STRING"},
    },
    "required": ["niche_compact_caption", "latent_steering_modifiers", "sdxl_dual_clip"],
}


# =============================================================================
# STAGE 0: INPUT LOADING
# =============================================================================

def _extract_style(instruction: str, fallback: str) -> str:
    """Pull a clean concept/style name out of an instruction.

    Handles the image v2 form ("...in the style of '<style>'") and the deforum form
    ("Generate a cinematic video prompt for: <concept>" / "Write a De Forum art film
    prompt for: <concept>").
    """
    m = re.search(r"style of '([^']+)'", instruction or "")
    if m:
        return m.group(1).strip().lstrip(":").strip()
    m = re.search(r"\bfor:\s*(.+)$", instruction or "", re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return (fallback or "").strip()


def load_inputs(args) -> tuple[list[dict], str]:
    """Return (rows, mode). mode ∈ {'transform', 'generate'}.

    TRANSFORM rows carry: concept, flux_seed, axis.
    GENERATE rows carry:  concept (the raw description).
    """
    raw: list[dict] = []

    if args.dataset:
        from datasets import load_dataset

        token = os.environ.get("HF_TOKEN")
        ds = load_dataset(args.dataset, split=args.split, token=token)
        raw = [dict(r) for r in ds]
    elif args.input:
        with open(args.input) as f:
            if args.input.endswith(".jsonl"):
                raw = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                raw = data if isinstance(data, list) else [data]
    else:
        text = sys.stdin.read().strip()
        if not text:
            return [], "generate"
        try:
            data = json.loads(text)
            raw = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            raw = [{"description": line} for line in text.splitlines() if line.strip()]

    if args.max_rows:
        raw = raw[: args.max_rows]

    # Auto-detect mode: a `response` column means we transform existing prompts.
    if raw and "response" in raw[0] and raw[0].get("response"):
        rows = []
        for r in raw:
            resp = (r.get("response") or "").strip()
            if not resp:
                continue
            rows.append({
                "concept": _extract_style(r.get("instruction", ""), r.get("style", "")),
                "flux_seed": resp,
                "axis": (r.get("axis") or "").strip(),
            })
        return rows, "transform"

    rows = []
    for r in raw:
        desc = (r.get("description") or r.get("concept") or r.get("text") or "").strip()
        if desc:
            rows.append({"concept": desc})
    return rows, "generate"


# =============================================================================
# STAGE 1: GENERATION (produces a 4-field bundle per concept)
# =============================================================================

def _make_user_prompt(row: dict, mode: str, dialect_set: str = "image") -> str:
    if mode == "transform":
        axis = f"\nEMPHASIS AXIS: {row['axis']}" if row.get("axis") else ""
        label = "SOURCE CINEMATIC PROMPT" if dialect_set == "video" else "SOURCE FLUX PROMPT"
        return (
            f"{label}:\n{row['flux_seed']}{axis}\n\n"
            "Derive the other prompt structures now."
        )
    label = "SCENE DESCRIPTION" if dialect_set == "video" else "IMAGE DESCRIPTION"
    return f"{label}:\n{row['concept']}\n\nWrite the prompt structures now."


def _extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction (Ollama may wrap output in fences/prose)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _assemble(row: dict, data: dict, mode: str, dialect_set: str = "image") -> dict | None:
    """Build a flat bundle (one key per dialect) from an LLM JSON payload.

    image: {concept, flux_t5, sdxl_dual_clip, compact_caption, steering_modifiers}
    video: {concept, wan_video, ltx_video, compact_caption}
    Returns None if any required field is missing.
    """
    if dialect_set == "video":
        # Video transform rewrites all three dialects (it does not anchor the source
        # verbatim — the source carries non-visual content that must be stripped).
        wan = (data.get("wan_video") or "").strip()
        ltx = (data.get("ltx_video") or "").strip()
        caption = (data.get("compact_caption") or "").strip()
        if not (wan and ltx and caption):
            return None
        return {
            "concept": row["concept"],
            "wan_video": wan,
            "ltx_video": ltx,
            "compact_caption": caption,
        }

    if mode == "transform":
        flux = row["flux_seed"]
        sdxl = (data.get("sdxl_dual_clip") or "").strip()
    else:
        routing = data.get("model_routing_prompts") or {}
        flux = (routing.get("flux_t5") or "").strip()
        sdxl = (routing.get("sdxl_dual_clip") or "").strip()

    caption = (data.get("niche_compact_caption") or "").strip()
    mods = (data.get("latent_steering_modifiers") or "").strip()
    if not (caption and mods and flux and sdxl):
        return None

    return {
        "concept": row["concept"],
        "flux_t5": flux,
        "sdxl_dual_clip": sdxl,
        "compact_caption": caption,
        "steering_modifiers": mods,
    }


def gemini_one(client, schema, row: dict, mode: str, dialect_set: str = "image") -> dict | None:
    from google.genai import types as genai_types

    system = SYSTEM_BY[(dialect_set, mode)]
    try:
        resp = client.models.generate_content(
            model=GEN_MODEL,
            contents=_make_user_prompt(row, mode, dialect_set),
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.8,
                top_p=0.9,
                max_output_tokens=4000,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        data = json.loads(resp.text)
        return _assemble(row, data, mode, dialect_set)
    except Exception as e:
        msg = str(e)[:160].replace("\n", " ")
        print(f"  ⚠ gen failed [{row['concept'][:40]!r}]: {type(e).__name__}: {msg}",
              file=sys.stderr)
        return None


def ollama_one(
    model: str, row: dict, mode: str, dialect_set: str = "image", debug: bool = False
) -> dict | None:
    import ollama as ollama_lib

    system = SYSTEM_BY[(dialect_set, mode)]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": _make_user_prompt(row, mode, dialect_set)},
    ]
    for attempt in range(3):
        try:
            resp = ollama_lib.chat(
                model=model, messages=messages,
                options={"temperature": 0.8, "top_p": 0.9},
            )
            content = (resp.message.content or "").strip()
            if debug:
                print(f"    [DEBUG] {content[:160]}", file=sys.stderr)
            data = _extract_json(content)
            if data:
                out = _assemble(row, data, mode, dialect_set)
                if out:
                    return out
            if attempt < 2:
                time.sleep(1)
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            print(f"  ⚠ ollama failed: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr)
            return None
    return None


def generate_all(rows: list[dict], mode: str, args) -> list[dict]:
    dset = args.dialect_set
    print(f"🎨 Stage 1: {mode} {len(rows)} concepts via {args.backend} (dialect-set={dset})")
    bundles: list[dict] = []

    if args.backend == "gemini":
        from google import genai

        client = genai.Client(api_key=args.gemini_key)
        schema = SCHEMA_BY[(dset, mode)]
        workers = 1 if args.concurrent <= 1 else min(args.concurrent, MAX_CONCURRENT)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(gemini_one, client, schema, r, mode, dset): r for r in rows
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="generate"):
                r = fut.result()
                if r is not None:
                    bundles.append(r)
    else:  # ollama — sequential (local model, no concurrency benefit)
        for row in tqdm(rows, desc="generate"):
            r = ollama_one(args.ollama_model, row, mode, dset, debug=args.debug)
            if r is not None:
                bundles.append(r)

    print(f"   {len(bundles)}/{len(rows)} bundles produced")
    return bundles


# =============================================================================
# STAGE 2: FAN-OUT + PER-DIALECT QUALITY GATES
# =============================================================================

def _words(text: str) -> int:
    return len(text.split())


def _has_bad_pattern(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE | re.MULTILINE) for p in BAD_PATTERNS)


def _gate_compact_caption(text: str) -> tuple[list[str], list[str]]:
    fails, warns = [], []
    cw = _words(text)
    if not (CAPTION_MIN_W <= cw <= CAPTION_MAX_W):
        fails.append(f"caption {cw}w outside [{CAPTION_MIN_W},{CAPTION_MAX_W}]")
    low = text.lower()
    if any(b in low for b in BUZZWORDS):
        fails.append("caption: filler buzzword")
    if not any(t in low for t in LIGHTING_TERMS):
        fails.append("caption: no lighting term")  # hard (85.5% pass in v2)
    if not any(t in low for t in CAMERA_TERMS):
        warns.append("caption: no camera term")     # soft — see calibration
    return fails, warns


def _gate_steering_modifiers(text: str) -> tuple[list[str], list[str]]:
    fails, warns = [], []
    tags = [t.strip() for t in text.split(",") if t.strip()]
    if not (MODIFIER_MIN <= len(tags) <= MODIFIER_MAX):
        fails.append(f"modifiers: {len(tags)} outside [{MODIFIER_MIN},{MODIFIER_MAX}]")
    if len(tags) != len({t.lower() for t in tags}):
        fails.append("modifiers: duplicate tags")
    if any(t.lower() in GENERIC_MODIFIERS for t in tags):
        fails.append("modifiers: generic term")
    if any(t != t.lower() for t in tags):
        warns.append("modifiers: not all lowercase")
    return fails, warns


def _gate_flux_t5(text: str) -> tuple[list[str], list[str]]:
    fails = []
    fw = _words(text)
    if not (FLUX_MIN_W <= fw <= FLUX_MAX_W):
        fails.append(f"flux_t5 {fw}w outside [{FLUX_MIN_W},{FLUX_MAX_W}]")
    if re.search(r"^\s*[-*•]|\n\s*\d+\.", text):
        fails.append("flux_t5: list formatting (not prose)")
    return fails, []


def _gate_sdxl_dual_clip(text: str) -> tuple[list[str], list[str]]:
    fails, warns = [], []
    sw = _words(text)
    if not (SDXL_MIN_W <= sw <= SDXL_MAX_W):
        fails.append(f"sdxl {sw}w outside [{SDXL_MIN_W},{SDXL_MAX_W}]")
    head = re.split(r"[\s,]+", text.strip().lower(), maxsplit=1)[0] if text.strip() else ""
    if head in SDXL_BOILERPLATE_HEADS:
        warns.append("sdxl: not front-loaded")
    return fails, warns


DIALECT_GATES = {
    "flux_t5": _gate_flux_t5,
    "sdxl_dual_clip": _gate_sdxl_dual_clip,
    "compact_caption": _gate_compact_caption,
    "steering_modifiers": _gate_steering_modifiers,
}


# =============================================================================
# VIDEO DIALECT SET (LTX-Video / WanVideo)
# =============================================================================
# Calibrated against Limbicnation/deforum-prompt-lora-dataset-v7 (1,547 train rows):
#   response words — median 48, p90 57, max 75 (0% reach 90w).
#   camera term present 89.2% (start with a camera clause 72.9%) → hard gate ok.
#   lighting term present 99.0% → hard gate ok.
#   non-visual token (scent/sound/memory/...) present 47.1% → strip in prompt + hard gate.
# The video corpus is cinematic prose, NOT image/lens prose, so camera/lighting vocab
# and word budgets differ from the image set above.

VIDEO_DIALECTS: dict[str, str] = {
    "wan_video": "Generate a WanVideo (UMT5) prompt for:",
    "ltx_video": "Generate an LTX-Video prompt for:",
    "compact_caption": "Write a compact video caption for:",
}

# Source responses are median 48w but ~47% of that is non-visual padding; once the
# rewrite strips scents/sounds/abstractions the visual core lands near 20-30w. A higher
# floor would force the model to re-pad with the fluff we are removing, so keep it at 20.
VID_MIN_W, VID_MAX_W = 20, 80
VCAPTION_MIN_W, VCAPTION_MAX_W = 4, 25  # plan §3: compact caption <=25 words

# Cinematic camera-motion vocabulary (not lens specs — see CAMERA_TERMS above for image).
VIDEO_CAMERA_TERMS = [
    "pan", "tilt", "zoom", "dolly", "track", "tracking", "crane", "push", "pull",
    "orbit", "aerial", "handheld", "static", "close-up", "closeup", "wide shot",
    "low-angle", "low angle", "high-angle", "high angle", "camera", "shot", "pov",
    "drone", "steadicam", "whip pan", "rack focus",
]
# Cinematic lighting/atmosphere vocabulary present in the deforum corpus.
VIDEO_LIGHTING_TERMS = [
    "light", "lighting", "lit", "chiaroscuro", "backlit", "back light", "rim light",
    "glow", "shadow", "silhouette", "grain", "film grain", "golden hour", "blue hour",
    "twilight", "dusk", "dawn", "candlelight", "neon", "volumetric", "god rays",
    "moonlight", "sunlight", "haze", "fog", "overcast",
]

# Non-visual content a camera cannot capture — abstractions, senses, sounds, emotions.
# Hard-rejected in every video dialect (defense in depth behind the LLM rewrite).
NONVISUAL_TERMS = [
    "scent", "aroma", "smell", "odor", "odour", "fragran", "perfume", "stench",
    "sound", "music", "melody", "song", "echo", "whisper", "hum", "silence",
    "scream", "sob", "cry", "voice", "noise", "roar", "rustle",
    "grief", "sorrow", "rebellion", "unspoken", "longing", "nostalg", "yearning",
    "memory", "memories", "remember", "dread", "despair", "loneliness",
    "taste", "flavor", "flavour",
]
NONVISUAL_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in NONVISUAL_TERMS) + r")\w*", re.IGNORECASE
)


def _has_nonvisual(text: str) -> bool:
    return bool(NONVISUAL_RE.search(text))


def _gate_wan_video(text: str) -> tuple[list[str], list[str]]:
    fails, warns = [], []
    w = _words(text)
    if not (VID_MIN_W <= w <= VID_MAX_W):
        fails.append(f"wan_video {w}w outside [{VID_MIN_W},{VID_MAX_W}]")
    low = text.lower()
    if not any(t in low for t in VIDEO_CAMERA_TERMS):
        fails.append("wan_video: no camera term")     # hard (89.2% pass in v7)
    if not any(t in low for t in VIDEO_LIGHTING_TERMS):
        fails.append("wan_video: no lighting term")   # hard (99.0% pass in v7)
    if _has_nonvisual(text):
        fails.append("wan_video: non-visual token")
    if re.search(r"^\s*[-*•]|\n\s*\d+\.", text):
        fails.append("wan_video: list formatting (not prose)")
    return fails, warns


def _gate_ltx_video(text: str) -> tuple[list[str], list[str]]:
    fails, warns = [], []
    w = _words(text)
    if not (VID_MIN_W <= w <= VID_MAX_W):
        fails.append(f"ltx_video {w}w outside [{VID_MIN_W},{VID_MAX_W}]")
    low = text.lower()
    if not any(t in low for t in VIDEO_CAMERA_TERMS):
        fails.append("ltx_video: no camera term")
    if _has_nonvisual(text):
        fails.append("ltx_video: non-visual token")
    if re.search(r"^\s*[-*•]|\n\s*\d+\.", text):
        fails.append("ltx_video: list formatting (not prose)")
    return fails, warns


def _gate_compact_caption_video(text: str) -> tuple[list[str], list[str]]:
    fails, warns = [], []
    w = _words(text)
    if not (VCAPTION_MIN_W <= w <= VCAPTION_MAX_W):
        fails.append(f"caption {w}w outside [{VCAPTION_MIN_W},{VCAPTION_MAX_W}]")
    low = text.lower()
    if any(b in low for b in BUZZWORDS):
        fails.append("caption: filler buzzword")
    if _has_nonvisual(text):
        fails.append("caption: non-visual token")
    if not any(t in low for t in VIDEO_LIGHTING_TERMS):
        warns.append("caption: no lighting term")     # soft — captions are terse
    return fails, warns


VIDEO_DIALECT_GATES = {
    "wan_video": _gate_wan_video,
    "ltx_video": _gate_ltx_video,
    "compact_caption": _gate_compact_caption_video,
}

_VIDEO_FIELD_RULES = (
    "STRICTLY REMOVE all non-visual content: scents, smells, tastes, sounds, music, "
    "silence, screams, voices, and abstract/emotional metaphors (memory, grief, "
    "rebellion, unspoken tension, longing). Keep ONLY what a camera can physically see.\n\n"
    "Field rules:\n"
    "- wan_video: 25-70 words. Order as Subject, then Scene, then Motion. Natural-language "
    "prose. Must include the camera movement and one concrete lighting/atmosphere cue.\n"
    "- ltx_video: 25-70 words. Motion-centric shot description: lead with the camera move, "
    "then subject and visible action. Prose, no bullet points, no lists.\n"
    "- compact_caption: 25 words or fewer. Dense global caption naming the subject, the "
    "setting, and one visible lighting cue. No filler adjectives."
)

SYSTEM_TRANSFORM_VIDEO = (
    "You are an expert prompt engineer for text-to-video diffusion models "
    "(WanVideo/UMT5, LTX-Video/T5-XXL). You are given an existing cinematic shot "
    "description. Rewrite it into video-model prompt dialects WITHOUT changing the "
    "subject, setting, camera movement, or lighting.\n\n"
    + _VIDEO_FIELD_RULES
    + "\n\nReturn STRICT JSON with keys: wan_video, ltx_video, compact_caption. "
    "No markdown fences, no preamble."
)

SYSTEM_GENERATE_VIDEO = (
    "You are an expert prompt engineer for text-to-video diffusion models "
    "(WanVideo/UMT5, LTX-Video/T5-XXL). Given a raw scene description, write three "
    "video-model prompt dialects for it.\n\n"
    + _VIDEO_FIELD_RULES
    + "\n\nReturn STRICT JSON with keys: wan_video, ltx_video, compact_caption. "
    "No markdown fences, no preamble."
)

VIDEO_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "wan_video": {"type": "STRING"},
        "ltx_video": {"type": "STRING"},
        "compact_caption": {"type": "STRING"},
    },
    "required": ["wan_video", "ltx_video", "compact_caption"],
}

SYSTEM_PROMPT_TRAIN_VIDEO = (
    "You are an expert text-to-video prompt generator for diffusion models "
    "(WanVideo, LTX-Video). Given a concept and a target model, produce the prompt in "
    "that model's dialect. No labels, no preamble, no command-line flags."
)

# Registries keyed by --dialect-set (and mode where relevant).
DIALECTS_BY = {"image": DIALECTS, "video": VIDEO_DIALECTS}
GATES_BY = {"image": DIALECT_GATES, "video": VIDEO_DIALECT_GATES}
SYSTEM_PROMPT_TRAIN_BY = {"image": SYSTEM_PROMPT_TRAIN, "video": SYSTEM_PROMPT_TRAIN_VIDEO}
SYSTEM_BY = {
    ("image", "transform"): SYSTEM_TRANSFORM,
    ("image", "generate"): SYSTEM_GENERATE,
    ("video", "transform"): SYSTEM_TRANSFORM_VIDEO,
    ("video", "generate"): SYSTEM_GENERATE_VIDEO,
}
SCHEMA_BY = {
    ("image", "transform"): TRANSFORM_SCHEMA,
    ("image", "generate"): GENERATE_SCHEMA,
    ("video", "transform"): VIDEO_SCHEMA,
    ("video", "generate"): VIDEO_SCHEMA,
}


def fan_out_and_gate(
    bundles: list[dict], dialect_set: str = "image", debug: bool = False
) -> list[dict]:
    """Explode each bundle into flat rows (one per dialect); gate each independently."""
    print("🧹 Stage 2: fan-out + per-dialect gates")
    dialects = DIALECTS_BY[dialect_set]
    gates = GATES_BY[dialect_set]
    rows: list[dict] = []
    n_warn = 0
    per_dialect = {d: 0 for d in dialects}

    for b in bundles:
        for dialect, prefix in dialects.items():
            text = (b.get(dialect) or "").strip()
            if not text:
                continue
            fails, warns = gates[dialect](text)
            if _has_bad_pattern(text):
                fails.append(f"{dialect}: forbidden pattern (flag/markdown/preamble)")
            if fails:
                if debug:
                    print(f"   ✗ {dialect} [{b['concept'][:30]!r}]: {'; '.join(fails)}",
                          file=sys.stderr)
                continue
            if warns:
                n_warn += 1
            rows.append({
                "original_concept": b["concept"],
                "target_model": dialect,
                "instruction": f"{prefix} {b['concept']}",
                "response": text,
            })
            per_dialect[dialect] += 1

    total_candidates = len(bundles) * len(dialects)
    print(f"   {len(rows)}/{total_candidates} rows passed ({n_warn} with soft warnings)")
    print("   by dialect: " + ", ".join(f"{d}={n}" for d, n in per_dialect.items()))
    return rows


# =============================================================================
# STAGE 3: OUTPUT
# =============================================================================

def render_text_field(rows: list[dict], dialect_set: str = "image") -> None:
    """Add a Qwen-chat-template `text` column in-place (drop-in for train_sft.py)."""
    from transformers import AutoTokenizer

    print(f"📝 Rendering `text` field with {TRAIN_TOKENIZER_ID}")
    system_prompt = SYSTEM_PROMPT_TRAIN_BY[dialect_set]
    tok = AutoTokenizer.from_pretrained(
        TRAIN_TOKENIZER_ID, token=os.environ.get("HF_TOKEN"), trust_remote_code=False
    )
    for r in rows:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": r["instruction"]},
            {"role": "assistant", "content": r["response"]},
        ]
        r["text"] = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )


def write_jsonl(records: list[dict], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"   💾 Wrote {len(records)} rows → {p}")


def push_dataset(records: list[dict], hub_id: str, token: str) -> None:
    from datasets import Dataset, DatasetDict

    print(f"📤 Stage 3: pushing {len(records)} rows to {hub_id}")
    ds = Dataset.from_list(records)
    split = ds.train_test_split(test_size=0.10, seed=SEED)
    try:
        DatasetDict({"train": split["train"], "validation": split["test"]}).push_to_hub(
            hub_id, private=True, token=token
        )
        print(f"   🎉 https://huggingface.co/datasets/{hub_id}")
    except Exception as e:
        print(f"   ⚠ Hub push failed: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        print("   Local JSONL backup is intact — upload separately once token has "
              "write permission.", file=sys.stderr)
        raise


# =============================================================================
# ORCHESTRATION
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--dataset", help="HF dataset id (transform mode if it has a `response` column)")
    src.add_argument("--input", help="Local JSONL/JSON file of raw descriptions")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument(
        "--dialect-set", choices=["image", "video"], default="image",
        help="image: flux_t5/sdxl/caption/modifiers; video: wan_video/ltx_video/caption",
    )
    parser.add_argument("--backend", choices=["gemini", "ollama"], default="gemini")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Local JSONL output path")
    parser.add_argument("--hub-id", help="Push result to this HF dataset id")
    parser.add_argument("--render-text", action="store_true",
                        help="Add Qwen-chat-template `text` column for train_sft.py")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit number of source concepts (not output rows)")
    parser.add_argument("--concurrent", type=int, default=MAX_CONCURRENT,
                        help="Gemini concurrency (ignored for ollama)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run all stages but skip file write and Hub push")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    args.gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if args.backend == "gemini" and not args.gemini_key:
        print("ERROR: set GEMINI_API_KEY (or GOOGLE_API_KEY) for --backend gemini",
              file=sys.stderr)
        return 1

    hf_token = os.environ.get("HF_TOKEN")
    if args.hub_id and not hf_token:
        print("ERROR: HF_TOKEN required for --hub-id push", file=sys.stderr)
        return 1

    print("🌱 Stage 0: loading inputs")
    rows, mode = load_inputs(args)
    if not rows:
        print("ERROR: no input rows found (use --dataset, --input, or pipe JSON to stdin)",
              file=sys.stderr)
        return 1
    print(f"   {len(rows)} concepts, mode={mode}")

    t0 = time.time()
    bundles = generate_all(rows, mode, args)
    final = fan_out_and_gate(bundles, dialect_set=args.dialect_set, debug=args.debug)

    print(f"\n📊 Summary (elapsed {time.time() - t0:.1f}s):")
    print(f"   concepts   → {len(rows)}")
    print(f"   bundles    → {len(bundles)}")
    print(f"   final rows → {len(final)}")

    if final:
        s = final[0]
        print("\n🔎 Sample row:")
        print(f"   target_model     : {s['target_model']}")
        print(f"   instruction      : {s['instruction'][:100]}")
        print(f"   response         : {s['response'][:160]}")

    if args.dry_run:
        print("\n✅ Dry run complete (no write, no push).")
        return 0

    if not final:
        print("ERROR: no rows survived the quality gate; nothing to write", file=sys.stderr)
        return 2

    # Write the local backup BEFORE rendering text — a render failure must never
    # discard generated rows (this previously lost a full run to a missing jinja2).
    write_jsonl(final, args.output)
    if args.render_text:
        render_text_field(final, dialect_set=args.dialect_set)
        write_jsonl(final, args.output)  # re-write with the `text` column
    if args.hub_id:
        push_dataset(final, args.hub_id, hf_token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
