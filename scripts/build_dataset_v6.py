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
#   "requests>=2.28.0",
#   "ollama>=0.4.0",
# ]
# ///
"""
Build Deforum Prompt Dataset v6 — Decoupled Instruction/Synthesis Sources

Key fix over v5:
- Stage 2/3 instructions now use a RANDOM SCENE SEED (from SCENE_SEEDS pool)
  instead of raw creative-writing / Gutenberg text as the {scene} variable.
- Source texts (creative writing, Gutenberg) are used ONLY internally by Ollama
  for synthesis diversity — they NEVER appear in the instruction the model trains on.
- This prevents the v5 failure: model outputting story prose / ShareGPT-style
  "write a story..." instructions when given a simple scene description.

All other v5 improvements retained:
- Hard-reject filters: SD params, screenplay markers, synthesis leakage, input echo
- 12 instruction templates sampled randomly (~half without "prompt" keyword)
- Post-filter validation: 50-row sample check, halt if >5% fail
- Word count range: 30-90

Sources:
  Stage 1: ~260 curated scene seeds (instruction = scene seed, synthesis = scene seed)
  Stage 2: Creative Writing ShareGPT (instruction = scene seed, synthesis = story passage)
  Stage 3: Gutenberg Sci-Fi (instruction = scene seed, synthesis = book chunk)

Usage:
    # Dry-run (5 samples per stage)
    conda run -n prompt-lora-trainer uv run scripts/build_dataset_v6.py --dry-run --seed 42

    # Pilot (~300 rows)
    conda run -n prompt-lora-trainer uv run scripts/build_dataset_v6.py \\
      --seed 42 --stage1-limit 100 --stage2-limit 100 --stage3-limit 100

    # Full run (~1,600+ rows)
    conda run -n prompt-lora-trainer uv run scripts/build_dataset_v6.py \\
      --seed 42 --target Limbicnation/deforum-prompt-lora-dataset-v6

    # Full run with larger model
    conda run -n prompt-lora-trainer uv run scripts/build_dataset_v6.py \\
      --seed 42 --ollama-model qwen3:8b --target Limbicnation/deforum-prompt-lora-dataset-v6
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import duckdb
import ollama as ollama_lib
import pandas as pd
from datasets import Dataset, DatasetDict
from thefuzz import fuzz
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_MESSAGE = (
    "You are a cinematic video prompt generator specializing in the De Forum Art Film aesthetic."
)

OLLAMA_MODEL = "qwen3:4b"
OLLAMA_OPTIONS = {
    "temperature": 0.85,
    "top_p": 0.9,
    "num_predict": 3000,  # generous budget: ~2500 thinking + ~500 response tokens
    "repeat_penalty": 1.3,
}

# v5 Synthesis prompt — simple, list-free, no "Requirements: a, b, c"
SYNTHESIS_PROMPT = """Write a cinematic video diffusion prompt for this scene.

{source_text}

40-70 words. Begin with camera movement. Varied lighting. Film grain. No technical specs. No character names. No introductory phrases. Output ONLY the prompt."""


# =============================================================================
# INSTRUCTION TEMPLATES (12 variants, ~half without "prompt" keyword)
# =============================================================================

INSTRUCTION_TEMPLATES = [
    # With "prompt" keyword
    "Generate a cinematic video prompt for: {scene}",
    "Write a De Forum art film prompt for: {scene}",
    "Cinematic video prompt: {scene}",
    "Video diffusion prompt for {scene}",
    "Create a cinematic prompt for this scene: {scene}",
    "De Forum aesthetic prompt for: {scene}",
    "Atmospheric video prompt: {scene}",
    # Without "prompt" — prevents keyword overfitting
    "{scene}",
    "Create a video of: {scene}",
    "Visualize this scene: {scene}",
    "{scene} — describe this cinematically",
    "Video concept: {scene}",
]


# =============================================================================
# CURATED SCENE SEEDS — ~260 hand-crafted seeds (Stage 1 source)
# No screenplay formatting, no SD technical params, pure scene descriptions.
# =============================================================================

SCENE_SEEDS = [
    # --- Urban: Rain & Night ---
    "rain-soaked alley at night, neon signs bleeding color into wet asphalt",
    "empty intersection at 3am, traffic lights cycling through in heavy rain",
    "underground parking structure, fluorescent light flickering over wet concrete",
    "fire escape descending into a flooded courtyard at midnight",
    "neon diner window at 2am, lone customer visible through condensation",
    "rain drumming on taxi roof, city lights smearing through fogged glass",
    "canal bridge at midnight, lamp reflection broken by rain rings",
    "wet rooftop with city skyline glowing below, figure at the edge",
    "service alley behind restaurant, steam vents and drizzle at 3am",
    "phone booth interior, rain hammering the glass, city lights blurred outside",
    "subway entrance at night, rain cascading down the steps",
    "soaked marketplace awning, water streaming off canvas in rivulets",
    "overpass at midnight, headlights below painting wet concrete gold",
    "laundromat window at night, rain on glass, machines spinning inside",
    "industrial canal with towpath, orange sodium lights in rain",
    "city bridge at 4am, wind-driven rain, a single umbrella crossing",
    "back street behind a bar, spilled light and cigarette smoke in mist",
    "rooftop water tower, rain and wind, city glittering below in storm",
    "narrow European alley in downpour, cobblestones glistening",
    "train overpass at night, rain falling between tracks, empty platform below",
    # --- Urban: Day & Dusk ---
    "busy crossroads at rush hour, pedestrians moving in blurred streams",
    "brutalist apartment facade, morning light raking across concrete grids",
    "elevated train platform, passengers waiting in early morning haze",
    "rooftop garden above the city, someone tending to plants at dawn",
    "street market at dawn, vendors arranging produce in half-light",
    "construction site at lunch, workers resting on exposed steel beams",
    "urban underpass at midday, filtered light through graffiti-tagged girders",
    "city square at golden hour, pigeons lifting in a swirl of light",
    "laundromat interior, morning light on spinning machines",
    "bakery window at first light, loaves stacked behind steamed glass",
    "harbour at sunrise, fishing boats returning through mist",
    "concrete park at dusk, children on swings, long shadows",
    "elevated highway at sunset, traffic moving in slow amber light",
    "courtyard apartment complex at midday, laundry lines crossing overhead",
    "fire station bay, gleaming truck, afternoon light angling through doors",
    "city rooftop at dusk, cooling towers and antenna forests silhouetted",
    "commuter train interior at evening, reflected faces in dark windows",
    "bus depot at end of day, empty buses parked in geometric rows",
    "underpass mural at golden hour, bicyclist passing through",
    "city park pond at twilight, lamp reflections stretching across water",
    # --- Urban: Industrial & Abandoned ---
    "abandoned factory floor, broken skylights letting in shafts of light",
    "derelict warehouse with pigeons roosting in the iron rafters",
    "rail yard at dusk, rusty freight cars and tall weeds",
    "industrial chimney stack against an overcast sky",
    "demolition site at sunset, dust cloud lit gold by low sun",
    "power substation at night, humming transformers in blue-white light",
    "empty grain silo, light filtering through holes in the corrugated wall",
    "ship graveyard at low tide, rusted hulls on mudflats",
    "printing plant interior, paper rolls stacked to ceiling in silence",
    "old textile mill at dawn, cobwebs catching window light",
    "decommissioned nuclear cooling tower interior, sky visible above",
    "water treatment plant, circular clarifiers in morning mist",
    "salvage yard at golden hour, crushed cars piled to the sky",
    "brick sewer tunnel, arched ceiling, water trickling through",
    "abandoned swimming pool, peeling tiles and dry basin",
    "machine shop at closing time, tools hung on pegboards, light dimming",
    "chemical plant at night, flare stacks burning orange against clouds",
    "empty cold storage warehouse, frost breath, rows of empty hooks",
    "bus graveyard, overgrown coaches in a field, windshields fogged",
    "steel mill at night, molten metal glow through observation slits",
    # --- Nature: Forest & Woodland ---
    "ancient forest at dawn, mist threading between old-growth trunks",
    "dense jungle canopy, shafts of light cutting to the mossy floor",
    "pine forest in winter snow, silence and blue shadow",
    "bamboo grove at midday, wind making light shimmer through stalks",
    "autumn forest floor, fallen leaves in amber and rust, no wind",
    "forest clearing at night, moonlight on dewy grass",
    "gnarled oak at the edge of a field, branches against storm clouds",
    "forest stream in spring, clear water over smooth stones",
    "redwood grove, cathedral height, ferns in the filtered light",
    "mangrove roots at low tide, water threading between dark tangles",
    "boreal forest after wildfire, charred trunks in white snow",
    "larch forest in autumn, gold needles drifting",
    "forest path in fog, silhouette of a deer stopping ahead",
    "cedar forest in rain, drops falling from bough to bough",
    "eucalyptus grove at sunset, peeling bark lit amber",
    "hazel coppice in early spring, catkins and pale green light",
    "spruce forest in a storm, canopy heaving, dark and turbulent",
    "birch forest in early winter, white trunks and bare branches",
    "rainforest understory at dusk, enormous fern fronds",
    "forest edge at sunrise, deer tracks in frost",
    # --- Nature: Water & Coast ---
    "stormy sea coast, waves breaking over black rocks",
    "calm lake at sunset, perfect mirror, one ripple from a fish",
    "waterfall in a tropical gorge, mist rising, light refracting",
    "tidal flats at low water, wader birds on distant sandbar",
    "river delta from above, branching channels in late light",
    "frozen river surface, snow on ice, no tracks",
    "sea cave at high tide, water sloshing in the dark interior",
    "ocean at pre-dawn, horizon faintly brightening, no land",
    "flood plain after rain, fields under brown water, lone tree",
    "mountain lake, glass-still at dawn, reflection of peaks",
    "rocky coast at low tide, tide pools and anemones in clear water",
    "glacier face, blue-white ice, calving crack echoing",
    "slow river through farmland at dusk, herons wading",
    "estuary at high tide, light silver on water and mudflat",
    "open ocean swell at sunset, no land in sight",
    "stream in a chalk valley, watercress and chalk boulders",
    "harbour mouth at dawn, buoys and navigation lights",
    "bog landscape, cotton grass and pewter sky",
    "flooded quarry at noon, turquoise water, white rock face",
    "sea cliff at sunset, gusting wind, seabirds soaring",
    # --- Nature: Mountain & Highland ---
    "mountain ridge at sunrise, cloud sea below, lone figure",
    "rocky canyon at midday, vertical walls and thin slice of sky",
    "highland moorland in late autumn, heather brown and bracken gold",
    "glacier valley, hanging ice walls, terminal moraine",
    "high pass in winter, wind-carved snow and thin blue sky",
    "volcanic crater interior, steam vents and grey ash",
    "scree slope at dusk, loose stones and fading light",
    "alpine meadow in summer, wildflowers and distant peaks",
    "narrow mountain gorge, river below, cliff walls close",
    "summit in clouds, wind and intermittent visibility",
    "dry riverbed in mountain desert, boulders bleached white",
    "plateau edge, sudden drop to valley below, haze in the distance",
    "cirque lake, rock walls rising on three sides, still water",
    "col between two peaks, wind tearing across the gap",
    "mountain hut at night, single window lit, stars and snow",
    "rock face seen from below, climber's silhouette far above",
    "highland burn in spate, amber water over mossy boulders",
    "lava field at sunset, hardened black rock, no vegetation",
    "mountain footpath in autumn, golden birch and rushing stream",
    "desert mesa at golden hour, red rock and long shadow",
    # --- Nature: Desert & Arid ---
    "sand dunes at golden hour, crescent ridges and shadow",
    "salt flat at midday, heat shimmer dissolving the horizon",
    "desert canyon at sunrise, layered sandstone in orange light",
    "dry wash through red rock desert, flash-flood debris",
    "dust storm approaching a lone road at dusk",
    "oasis at noon, date palms and still water in white light",
    "stone desert, hamada, no sand, just black gravel to horizon",
    "dried lake bed, cracked mud hexagons in bleached light",
    "desert highway at night, headlights vanishing into darkness",
    "saguaro cactus forest at sunset, silhouettes against orange sky",
    "desert wadi after rain, running red water over baked earth",
    "badlands formation at golden hour, eroded spires and gullies",
    "sand sea from a ridge, dune after dune to the horizon",
    "desert ruins at dusk, crumbling mud walls and long shadow",
    "dry rocky plateau, nothing moves, heat mirages",
    "ancient desert trade route, ruts worn in stone over centuries",
    "atacama at night, clarity of stars over bare rock",
    "desert sandstone arch, valley visible through the frame",
    "camel track in clean sand, single row of prints vanishing",
    "desert thunderstorm, lightning on the mesa, rain not reaching ground",
    # --- Interior: Library & Archive ---
    "vast library with iron spiral staircases, books to the ceiling",
    "dimly lit reading room, pools of lamplight on green baize",
    "card catalogue room, wooden drawers and brass handles in amber light",
    "manuscript archive, white-gloved archivist, fragile pages",
    "attic library, skylight, dust motes, stacked boxes",
    "private study at midnight, firelight and leather spines",
    "empty lecture hall, tiered seats and a single projected slide",
    "map room, rolled charts and flat-file cabinets, window light",
    "library stacks at closing time, lights going out row by row",
    "reading room at dawn, first light on long oak tables, no one yet",
    "antiquarian bookshop at dusk, narrow passages between floor-to-ceiling shelves",
    "rare books room, humidity control unit humming, glass cases",
    "scriptorum in a monastery, high windows, quill and vellum",
    "newspaper archive, microfiche readers, fluorescent strip lights",
    "university reading room, vaulted ceiling and gallery rail",
    # --- Interior: Warehouse & Industrial ---
    "vast warehouse under skylights, forklift tracks in dust",
    "cold storage chamber, breath visible, racks to the ceiling",
    "abandoned factory lunchroom, peeling posters, overturned tables",
    "boiler room, copper pipes sweating, gauges and valves",
    "underground parking lot, low ceiling, concrete pillars, silence",
    "cargo hold of a ship, metal walls, crates in darkness",
    "aircraft hangar interior, single jet under work lights",
    "textile warehouse, bolts of fabric floor to ceiling, muted colours",
    "wine cellar, barrels in rows, dusty bottles on stone shelves",
    "museum storage room, covered sculptures and crated paintings",
    "data centre corridor at night, blinking lights and server hum",
    "brewery fermentation room, tanks and CO2 haze",
    "old mill interior, millstone and grain dust in shaft of light",
    "printing works at shift change, machines slowing, lights dimming",
    "cold chain warehouse, pallets of frozen goods, ice-crystal air",
    # --- Interior: Station & Transit ---
    "grand railway terminus at night, vaulted ceiling and empty platforms",
    "subway tunnel, approaching train headlights, rush of air",
    "airport terminal at 5am, cleaner and a sleeping traveller",
    "harbour waiting room, ferry not yet arrived, rain on windows",
    "bus depot at midnight, driver checking timetable under fluorescent lights",
    "metro station, platform from the end, tunnel receding",
    "train carriage at night, dark landscape and reflected interior",
    "empty departure hall, boarding gates closed, dawn coming",
    "ferry vehicle deck, cars strapped down, sea visible through open stern",
    "underground concourse, turnstiles and rush-hour echoes fading",
    "taxi rank at 3am, two cabs, rain on tarmac",
    "canal lock, water rising, stone walls and iron gates",
    "lighthouse interior, spiral stair ascending toward the lens",
    "old signal box, levers and windows onto curved rail line",
    "port container terminal at night, cranes over stacked steel boxes",
    # --- Interior: Medical & Scientific ---
    "hospital corridor at 3am, distant trolley sound, strip lights",
    "operating theatre empty after procedure, light still on",
    "research laboratory, white coats and centrifuge hum",
    "morgue viewing room, tile and cold light through frosted glass",
    "abandoned sanatorium dayroom, overturned chairs and peeling paint",
    "pharmacy dispensary at night, row of bottles and single working lamp",
    "archive of specimen jars, amber and pale, on metal shelves",
    "MRI scanner in an empty room, magnet hum, no patient",
    "university physics lab, equations on a blackboard, instruments",
    "observatory dome interior, telescope and slit of night sky",
    # --- Abstract / Mood: Solitude ---
    "single figure in a long corridor, end not visible",
    "lone person standing at the edge of a pier at dusk",
    "empty dining table set for one, window with night outside",
    "rocking chair on a porch, no one in it, evening settling",
    "single candle in a dark room, everything else in shadow",
    "bed unmade in first light, person gone but impression remains",
    "coat on a hook in an empty entrance hall",
    "figure at an upper window, watching rain on the street below",
    "park bench in fog, footprints approaching but no one there",
    "empty swing still moving, no child, dusk",
    "single light in an otherwise dark apartment building",
    "figure on a platform as the last train pulls away",
    "empty classroom, chalk equations still on the board",
    "lone tree in a flat field, long shadow, golden hour",
    "smoke from a chimney in still morning air, no other movement",
    # --- Abstract / Mood: Contemplation & Time ---
    "hands cupped around a hot drink in cold morning light",
    "dust motes in a sunbeam through old glass, nothing moves",
    "clock face in an antique shop, multiple ticks, no alignment",
    "shadow moving slowly across a white wall as the day passes",
    "reflection in a window: inside warmth, outside winter street",
    "fire dying in a grate, embers pulsing, room darkening",
    "snow falling on a still, dark garden at night",
    "footsteps in sand being erased by a wave",
    "frost forming on a window pane, slow crystalline advance",
    "a door ajar, light inside, footsteps fading away",
    "rain on a tin roof at night, close interior, low light",
    "candlelight in an empty church nave, echo of footsteps",
    "old map on a table, finger tracing a route no longer travelled",
    "newspaper left on a bench, wind riffling pages, park empty",
    "autumn leaves falling past a lit window from below",
    "tide mark on stone wall of a harbour, history of water levels",
    "sunlight crossing a floor over the course of an afternoon",
    "empty hourglass, last grain fallen, no one watching",
    "candle stub burning out in an empty room",
    "morning mist on a still pond, lifting slowly",
    # --- Sci-Fi & Speculative ---
    "space station observation window, Earth rotating slowly below",
    "derelict spacecraft interior, hull breach, stars visible",
    "orbital dock at night, servicing pods and docking lights",
    "deep-space corridor, running lights only, no crew visible",
    "cyberpunk street market, neon and rain, augmented vendors",
    "underground megacity, tiers of light descending to haze below",
    "holographic city map in a dark command room",
    "android maintenance bay, units charging in rows, dim blue light",
    "nuclear reactor control room, pre-shutdown, warning lights amber",
    "server farm catacombs, endless blinking LED rows",
    "greenhouse module in a remote facility, barren landscape visible through dome",
    "orbital debris field, slow tumbling wreckage in sunlight",
    "ocean floor research station, porthole view of abyss",
    "arctic research base in storm, instruments and whiteout",
    "communications array in desert at night, dish aimed at stars",
    "cryogenic storage bay, frost on pods, minimal lighting",
    "post-collapse city overgrown with vegetation, towers reclaimed",
    "underwater tunnel flooded, fish swimming through old corridors",
    "empty lunar surface, rover tracks in regolith, Earth visible",
    "launch control room, countdown on screens, empty chairs",
]


# =============================================================================
# HARD-REJECT QUALITY GATE PATTERNS
# =============================================================================

SD_PARAM_PATTERNS = [
    r"--ar\s+\d+",
    r"--model\s+\w",
    r"--seed\s+\d+",
    r"\b4K\b",
    r"\b8K\b",
    r"\b1080p\b",
    r"\b720p\b",
    r"\b\d+fps\b",
    r"\bfps\b",
    r"\bHDR\b",
    r"\bmotion blur\b",
    r"\bshallow depth of field\b",
    r"\bdynamic range\b",
]

SCREENPLAY_PATTERNS = [
    r"(?i)^Scene\s+\d+\s*:",
    r"(?i)^Aspect Ratio\s*:",
    r"(?i)^Camera\s*:\s+",
    r"(?i)^Movement\s*:\s+",
    r"(?i)^Lighting\s*:\s+",
    r"(?i)^Film Style\s*:",
    r"(?i)^Mood\s*:\s+",
]

SYNTHESIS_LEAKAGE_PATTERNS = [
    r"(?i)include the following elements?",
    r"(?i)requirements?\s*:",
    r"(?m)^[\d]+[.)]\s+\w",   # ordered list: "1. something" or "1) something"
    r"(?m)^[a-z][.)]\s+\w",   # alpha list: "a. something" or "a) something"
    r"(?i)^here'?s",
    r"(?i)^certainly",
    r"(?i)^sure[.,!\s]",
    r"(?i)^i'?d be happy",
]

# All reject patterns combined for validation checks
ALL_REJECT_PATTERNS = SD_PARAM_PATTERNS + SCREENPLAY_PATTERNS + SYNTHESIS_LEAKAGE_PATTERNS

CHARACTER_NAME_PATTERN = (
    r"\b(?:Sarah|John|Elena|Mara|Yuki|David|James|Michael|Emma|Alice|Bob|Maria|Anna)\b"
)

# Visual/atmospheric scoring keywords
SCORING_KEYWORDS = {
    "visual": [
        "light", "shadow", "glow", "shimmer", "dark", "bright", "haze", "fog",
        "smoke", "dust", "flame", "neon", "silhouette", "reflection", "beam",
        "radiance", "gleam", "flicker", "pulse", "luminescence",
    ],
    "atmospheric": [
        "silence", "whisper", "echo", "wind", "rain", "thunder", "creak",
        "hum", "breathe", "stillness", "quiet", "hush", "murmur",
    ],
    "texture": [
        "grain", "rough", "smooth", "cold", "warm", "damp", "velvet", "rust",
        "glass", "metal", "concrete", "fabric",
    ],
    "movement": [
        "drift", "float", "crawl", "sweep", "cascade", "ripple", "flicker",
        "sway", "surge", "flow", "glide", "linger",
    ],
    "scifi": [
        "hologram", "viewport", "console", "starfield", "nebula", "reactor",
        "dome", "corridor", "airlock", "hull", "cryo", "terminal", "interface",
    ],
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def count_words(text: str) -> int:
    return len(text.split())


def score_text_for_visual_density(text: str, include_scifi: bool = False) -> float:
    text_lower = text.lower()
    score = 0
    for category, keywords in SCORING_KEYWORDS.items():
        if category == "scifi" and not include_scifi:
            continue
        for keyword in keywords:
            score += text_lower.count(keyword)
    word_count = count_words(text)
    if word_count > 0:
        score = score / (word_count ** 0.5)
    return score


def assign_tier(word_count: int) -> str:
    if word_count < 40:
        return "short"
    elif word_count < 70:
        return "medium"
    else:
        return "detailed"


def format_chat_template(instruction: str, response: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )


def random_instruction(rng: random.Random, scene: str) -> str:
    """Pick a random instruction template and format with scene description."""
    template = rng.choice(INSTRUCTION_TEMPLATES)
    # Truncate scene to avoid very long instructions
    scene = scene[:150].rstrip(",. ")
    return template.format(scene=scene)


def extract_best_paragraph(text: str, min_words: int = 40, max_words: int = 80) -> str:
    """Extract the highest-scoring paragraph in the 40-80 word range."""
    candidates = []

    for block in re.split(r"\n\n+", text):
        block = block.strip()
        if not block:
            continue
        words = block.split()
        wc = len(words)

        if min_words <= wc <= max_words:
            candidates.append(block)
        elif wc > max_words:
            # Slide a sentence window to find 40-80 word excerpts
            sentences = re.split(r"(?<=[.!?])\s+", block)
            window: List[str] = []
            for sent in sentences:
                window.append(sent)
                window_text = " ".join(window)
                if len(window_text.split()) > max_words:
                    window = window[1:]  # drop oldest sentence
                if min_words <= len(" ".join(window).split()) <= max_words:
                    candidates.append(" ".join(window))

    if not candidates:
        # Fallback: first 60 words
        return " ".join(text.split()[:60])

    scored = sorted(candidates, key=lambda p: score_text_for_visual_density(p), reverse=True)
    return scored[0]


def clean_ollama_response(text: str) -> str:
    """Strip thinking tokens, quotes, word-count annotations, prompt prefixes."""
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Handle think=False: thinking output ends with </think> without opening tag
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    # Remove surrounding quotes
    text = text.strip().strip('"').strip("'").strip()

    # Remove "**Prompt:**" or "Prompt:" prefix
    text = re.sub(
        r"^\s*\**\s*(?:Prompt|Output)\s*\d*\s*:?\s*\**\s*", "", text, flags=re.IGNORECASE
    )

    # Remove trailing word count annotations: "(75 words)" or "— 72 words"
    text = re.sub(r"\s*[\(\[—–-]\s*\d+\s*words?\s*[\)\]]?\s*$", "", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_input_echo(response: str, instruction: str, threshold: float = 0.5) -> bool:
    """Return True if >threshold of response words appear in instruction (input echo)."""
    response_words = set(response.lower().split())
    instruction_words = set(instruction.lower().split())
    if not response_words:
        return False
    overlap = response_words & instruction_words
    return len(overlap) / len(response_words) > threshold


def apply_quality_gate(response: str, instruction: str) -> Optional[str]:
    """Return rejection reason string, or None if response passes all gates."""
    # 1. Word count: 30-90
    wc = count_words(response)
    if wc < 30:
        return f"too_short:{wc}"
    if wc > 90:
        return f"too_long:{wc}"

    # 2. SD param patterns
    for pattern in SD_PARAM_PATTERNS:
        if re.search(pattern, response):
            return f"sd_param:{pattern}"

    # 3. Screenplay markers
    for pattern in SCREENPLAY_PATTERNS:
        if re.search(pattern, response, re.MULTILINE):
            return f"screenplay:{pattern}"

    # 4. Synthesis leakage
    for pattern in SYNTHESIS_LEAKAGE_PATTERNS:
        if re.search(pattern, response, re.MULTILINE):
            return f"leakage:{pattern}"

    # 5. Input echo
    if has_input_echo(response, instruction):
        return "input_echo"

    # 6. Character names
    if re.search(CHARACTER_NAME_PATTERN, response, re.IGNORECASE):
        return "character_name"

    # 7. Repeated phrases (5-gram, max 2)
    words = response.lower().split()
    if len(words) >= 5:
        phrases: Dict[str, int] = {}
        for i in range(len(words) - 4):
            phrase = " ".join(words[i : i + 5])
            phrases[phrase] = phrases.get(phrase, 0) + 1
            if phrases[phrase] > 2:
                return "repeated_phrase"

    return None  # passes all gates


# =============================================================================
# OLLAMA SYNTHESIS
# =============================================================================

def ollama_generate(
    source_text: str,
    model: str = None,
    retries: int = 2,
    debug: bool = False,
    scene_seed: Optional[str] = None,
) -> Optional[str]:
    """Synthesize a cinematic prompt from source text using Ollama.

    If scene_seed is provided (Stage 2/3), the prompt instructs Ollama to write
    about the scene seed while drawing visual style from source_text.
    Uses ollama.chat() to properly separate thinking tokens from content.
    """
    model = model or OLLAMA_MODEL
    if scene_seed:
        # Stage 2/3: scene seed drives the topic; source text provides style/texture
        prompt = (
            f"Write a cinematic video diffusion prompt for this scene: {scene_seed}\n\n"
            f"Draw visual atmosphere and texture from this passage: {source_text[:300]}\n\n"
            "40-70 words. Begin with camera movement. Film grain. No technical specs. "
            "No character names. No introductory phrases. Output ONLY the prompt."
        )
    else:
        prompt = SYNTHESIS_PROMPT.format(source_text=source_text[:400])
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(retries + 1):
        try:
            resp = ollama_lib.chat(
                model=model,
                messages=messages,
                options=OLLAMA_OPTIONS,
            )
            content = (resp.message.content or "").strip()
            cleaned = clean_ollama_response(content)

            if debug:
                print(f"    [DEBUG] cleaned ({count_words(cleaned)} words): {cleaned[:200]}")

            wc = count_words(cleaned)
            if wc < 15:
                if debug:
                    print(f"    [DEBUG] REJECT too_short ({wc})")
                if attempt < retries:
                    time.sleep(1)
                    continue
                return None

            if wc > 120:
                # Hard truncate at 90 words
                cleaned = " ".join(cleaned.split()[:90]) + "."

            return cleaned

        except Exception as e:
            if attempt < retries:
                time.sleep(2)
                continue
            print(f"  ⚠️  Ollama error: {e}")
            return None

    return None


# =============================================================================
# STAGE 1: Curated Scene Seeds
# =============================================================================

def stage1_scene_seeds(
    rng: random.Random,
    limit: int,
    dry_run: bool = False,
    model: str = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Generate from curated SCENE_SEEDS. Resamples with replacement if limit > seed count."""
    print("\n" + "=" * 60)
    print("STAGE 1: Curated Scene Seeds")
    print(f"  Seeds available: {len(SCENE_SEEDS)}, limit: {limit}")
    print("=" * 60)

    seeds = list(SCENE_SEEDS)
    if limit > len(seeds):
        # Sample with replacement to reach target
        extra = rng.choices(seeds, k=limit - len(seeds))
        seeds = seeds + extra
    elif limit < len(seeds):
        seeds = rng.sample(seeds, limit)

    rng.shuffle(seeds)

    if dry_run:
        seeds = seeds[:5]
        print(f"  [DRY RUN] Using {len(seeds)} seeds")

    rows = []
    rejected = 0

    for seed in tqdm(seeds, desc="Stage 1: Synthesizing"):
        instruction = random_instruction(rng, seed)
        response = ollama_generate(seed, model=model, debug=debug)

        if response is None:
            rejected += 1
            continue

        reason = apply_quality_gate(response, instruction)
        if reason:
            if debug:
                print(f"  GATE ({reason}): {response[:80]}")
            rejected += 1
            continue

        word_count = count_words(response)
        rows.append({
            "instruction": instruction,
            "response": response,
            "tier": assign_tier(word_count),
            "word_count": word_count,
            "text": format_chat_template(instruction, response),
            "source": "scene_seeds",
        })

    result_df = pd.DataFrame(rows)
    print(f"\nStage 1: {len(result_df)} kept, {rejected} rejected")
    if len(result_df) > 0:
        print(f"  Tier dist: {result_df['tier'].value_counts().to_dict()}")
        print(f"  Word count: mean={result_df['word_count'].mean():.0f}, "
              f"min={result_df['word_count'].min()}, max={result_df['word_count'].max()}")

    return result_df


# =============================================================================
# STAGE 2: Creative Writing ShareGPT
# =============================================================================

def stage2_creative_writing(
    rng: random.Random,
    limit: int = 700,
    dry_run: bool = False,
    model: str = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Score Creative Writing passages and synthesize from best visual paragraph."""
    print("\n" + "=" * 60)
    print("STAGE 2: Creative Writing ShareGPT")
    print("=" * 60)

    conn = duckdb.connect()
    if HF_TOKEN:
        conn.execute("CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN ?);", [HF_TOKEN])

    path = "hf://datasets/ChaoticNeutrals/Creative_Writing-ShareGPT@~parquet/default/train/*.parquet"
    print(f"Loading from: {path}")

    try:
        df = conn.execute(f"SELECT * FROM read_parquet('{path}')").fetchdf()
    except Exception as e:
        print(f"Error loading Creative Writing dataset: {e}")
        conn.close()
        return pd.DataFrame()

    conn.close()
    print(f"Loaded {len(df)} conversations")

    # Extract and score
    scored_items = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        conversations = row.get("conversations", [])
        if isinstance(conversations, str):
            try:
                conversations = json.loads(conversations)
            except Exception:
                continue

        if conversations is None or not hasattr(conversations, "__len__") or len(conversations) < 2:
            continue

        human_msg = None
        gpt_msg = None
        for msg in conversations:
            if not isinstance(msg, dict):
                continue
            from_field = msg.get("from", "")
            value = msg.get("value", "")
            if from_field in ("human", "user") and not human_msg:
                human_msg = value
            elif from_field in ("gpt", "assistant") and not gpt_msg:
                gpt_msg = value
            if human_msg and gpt_msg:
                break

        if not human_msg or not gpt_msg or count_words(gpt_msg) < 30:
            continue

        score = score_text_for_visual_density(gpt_msg)
        scored_items.append({"human": human_msg, "gpt": gpt_msg, "score": score})

    print(f"Extracted {len(scored_items)} valid conversations")

    # Sort and take top 20%
    scored_items.sort(key=lambda x: x["score"], reverse=True)
    top_count = max(int(len(scored_items) * 0.2), 50)
    top_items = scored_items[:top_count]

    if len(top_items) > limit:
        top_items = rng.sample(top_items, limit)
    print(f"Selected {len(top_items)} for synthesis")

    if dry_run:
        top_items = top_items[:5]
        print(f"  [DRY RUN] {len(top_items)} samples")

    rows = []
    rejected = 0

    for item in tqdm(top_items, desc="Stage 2: Synthesizing"):
        # Use best 40-80 word visual paragraph as source text for Ollama synthesis
        source_text = extract_best_paragraph(item["gpt"], min_words=40, max_words=80)

        # v6 FIX: instruction uses a clean scene seed — NOT the story text.
        # Source text is only used internally by Ollama for style/texture inspiration.
        scene = rng.choice(list(SCENE_SEEDS))
        instruction = random_instruction(rng, scene)

        response = ollama_generate(source_text, model=model, debug=debug, scene_seed=scene)
        if response is None:
            rejected += 1
            continue

        reason = apply_quality_gate(response, instruction)
        if reason:
            if debug:
                print(f"  GATE ({reason}): {response[:80]}")
            rejected += 1
            continue

        word_count = count_words(response)
        rows.append({
            "instruction": instruction,
            "response": response,
            "tier": assign_tier(word_count),
            "word_count": word_count,
            "text": format_chat_template(instruction, response),
            "source": "creative_writing",
        })

    result_df = pd.DataFrame(rows)
    print(f"\nStage 2: {len(result_df)} kept, {rejected} rejected")
    if len(result_df) > 0:
        print(f"  Tier dist: {result_df['tier'].value_counts().to_dict()}")
        print(f"  Word count: mean={result_df['word_count'].mean():.0f}, "
              f"min={result_df['word_count'].min()}, max={result_df['word_count'].max()}")

    return result_df


# =============================================================================
# STAGE 3: Gutenberg Sci-Fi
# =============================================================================

def stage3_gutenberg(
    rng: random.Random,
    limit: int = 700,
    dry_run: bool = False,
    model: str = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Chunk and score Gutenberg Sci-Fi books, synthesize from best visual chunk."""
    print("\n" + "=" * 60)
    print("STAGE 3: Gutenberg Sci-Fi")
    print("=" * 60)

    conn = duckdb.connect()
    if HF_TOKEN:
        conn.execute("CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN ?);", [HF_TOKEN])

    path = "hf://datasets/stevez80/Sci-Fi-Books-gutenberg@~parquet/default/train/*.parquet"
    print(f"Loading from: {path}")

    try:
        df = conn.execute(f"SELECT * FROM read_parquet('{path}') USING SAMPLE 500").fetchdf()
    except Exception as e:
        print(f"Error loading Gutenberg dataset: {e}")
        conn.close()
        return pd.DataFrame()

    conn.close()
    print(f"Loaded {len(df)} books")

    # Chunk into ~300-word segments (cap at 5 chunks per book)
    chunks: List[Dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        raw = row.get("text", "")
        # Handle bytes fields (Gutenberg parquet stores text as bytes in some rows)
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="replace")
        else:
            text = str(raw)

        title = str(row.get("title", "Unknown"))

        # Strip UTF-8 BOM if present
        text = text.lstrip("\ufeff")

        # Strip Gutenberg header/footer (*** START OF ... *** blocks)
        text = re.sub(r"\*\*\*\s*START OF.*?\*\*\*", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"\*\*\*\s*END OF.*?\*\*\*", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"End of Project Gutenberg.*", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Strip leading boilerplate lines (Project Gutenberg header, distribution notice)
        text = re.sub(
            r"^(?:The Project Gutenberg eBook.*?\n|This ebook is for.*?\n|"
            r"Title:.*?\n|Author:.*?\n|Release date:.*?\n|Language:.*?\n|"
            r"Produced by.*?\n|\s*\r?\n)+",
            "",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        words = text.split()
        book_chunks = 0
        i = 0
        while i < len(words) and book_chunks < 5:
            chunk_words = words[i : i + 350]
            if len(chunk_words) < 100:
                break
            chunk_text = " ".join(chunk_words)
            if 200 <= len(chunk_words) <= 500:
                chunks.append({"text": chunk_text, "title": title})
                book_chunks += 1
            i += 350

    print(f"Created {len(chunks)} chunks")

    # Score and take top
    for chunk in tqdm(chunks, desc="Scoring"):
        chunk["score"] = score_text_for_visual_density(chunk["text"], include_scifi=True)

    chunks.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = chunks[: limit * 3]  # extra buffer for synthesis failures

    if len(top_chunks) > limit:
        top_chunks = rng.sample(top_chunks, limit)
    print(f"Selected {len(top_chunks)} chunks for synthesis")

    if dry_run:
        top_chunks = top_chunks[:5]
        print(f"  [DRY RUN] {len(top_chunks)} samples")

    rows = []
    rejected = 0

    for chunk in tqdm(top_chunks, desc="Stage 3: Synthesizing"):
        source_text = chunk["text"]

        # v6 FIX: instruction uses a clean scene seed — NOT the Gutenberg text.
        # Source text is only used internally by Ollama for style/texture inspiration.
        scene = rng.choice(list(SCENE_SEEDS))
        instruction = random_instruction(rng, scene)

        response = ollama_generate(source_text, model=model, debug=debug, scene_seed=scene)
        if response is None:
            rejected += 1
            continue

        reason = apply_quality_gate(response, instruction)
        if reason:
            if debug:
                print(f"  GATE ({reason}): {response[:80]}")
            rejected += 1
            continue

        word_count = count_words(response)
        rows.append({
            "instruction": instruction,
            "response": response,
            "tier": assign_tier(word_count),
            "word_count": word_count,
            "text": format_chat_template(instruction, response),
            "source": "gutenberg_scifi",
        })

    result_df = pd.DataFrame(rows)
    print(f"\nStage 3: {len(result_df)} kept, {rejected} rejected")
    if len(result_df) > 0:
        print(f"  Tier dist: {result_df['tier'].value_counts().to_dict()}")
        print(f"  Word count: mean={result_df['word_count'].mean():.0f}, "
              f"min={result_df['word_count'].min()}, max={result_df['word_count'].max()}")

    return result_df


# =============================================================================
# STAGE 4: Quality Filter + Dedup + Validate + Push
# =============================================================================

def compute_hash(text: str) -> str:
    return hashlib.md5(text[:40].lower().encode()).hexdigest()[:8]


def fuzzy_deduplicate(df: pd.DataFrame, threshold: int = 85) -> pd.DataFrame:
    """Fuzzy deduplicate responses using thefuzz token_sort_ratio."""
    print(f"\nDeduplicating {len(df)} rows (threshold={threshold})...")
    df = df.copy()
    df["_hash"] = df["response"].apply(compute_hash)

    keep_indices = []
    for (_, _hash), group in tqdm(df.groupby(["tier", "_hash"]), desc="Deduplicating"):
        if len(group) <= 1:
            keep_indices.extend(group.index.tolist())
            continue

        responses = group["response"].tolist()
        indices = group.index.tolist()
        skip: set = set()
        for i in range(len(responses)):
            if i in skip:
                continue
            keep_indices.append(indices[i])
            for j in range(i + 1, len(responses)):
                if j in skip:
                    continue
                if fuzz.token_sort_ratio(responses[i], responses[j]) >= threshold:
                    skip.add(j)

    result = df.loc[keep_indices].drop(columns=["_hash"])
    print(f"Deduplication: {len(df)} → {len(result)} ({len(df) - len(result)} removed)")
    return result.reset_index(drop=True)


def post_filter_validation(
    df: pd.DataFrame, sample_size: int = 50, fail_threshold: float = 0.05
) -> bool:
    """
    Sample rows and run ALL reject patterns.
    Return True (halt) if >fail_threshold of samples fail; False (OK) otherwise.
    """
    print("\n" + "=" * 60)
    print("POST-FILTER VALIDATION SAMPLING")
    print("=" * 60)

    sample = df.sample(min(sample_size, len(df)), random_state=42)
    failures = []

    for idx, row in sample.iterrows():
        response = row["response"]
        instruction = row["instruction"]
        for pattern in ALL_REJECT_PATTERNS:
            if re.search(pattern, response, re.MULTILINE):
                failures.append((idx, pattern, response[:100]))
                break
        else:
            # Also check input echo
            if has_input_echo(response, instruction):
                failures.append((idx, "input_echo", response[:100]))

    fail_rate = len(failures) / len(sample)
    print(f"Sample: {len(sample)} rows, failures: {len(failures)} ({100 * fail_rate:.1f}%)")

    if failures:
        print("\n  Failed samples:")
        for idx, pattern, text in failures[:5]:
            print(f"    pattern: {pattern!r}")
            print(f"    text:    {text!r}")

    if fail_rate > fail_threshold:
        print(f"\n⚠️  HALT: {100 * fail_rate:.1f}% exceeds {100 * fail_threshold:.1f}% threshold")
        print("   Review and rerun with stricter synthesis prompt or more seeds.")
        return True  # halt

    print(f"✅ Validation passed ({100 * fail_rate:.1f}% < {100 * fail_threshold:.1f}%)")
    return False  # OK


def final_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Run comprehensive quality filter on combined dataframe."""
    print(f"\nFinal quality filter on {len(df)} rows...")
    original_len = len(df)

    # Word count
    df = df[(df["word_count"] >= 30) & (df["word_count"] <= 90)].copy()
    print(f"  Word count (30-90): {len(df)} ({original_len - len(df)} removed)")
    after_wc = len(df)

    # All hard-reject patterns
    def row_fails_gate(row: pd.Series) -> bool:
        return apply_quality_gate(row["response"], row["instruction"]) is not None

    mask = df.apply(row_fails_gate, axis=1)
    df = df[~mask]
    print(f"  Quality gate: {len(df)} ({after_wc - len(df)} removed)")

    print(f"\nFinal filter total: {original_len} → {len(df)} ({original_len - len(df)} removed)")
    return df.reset_index(drop=True)


def validate_and_sample(df: pd.DataFrame) -> None:
    """Print validation stats and sample outputs."""
    print("\n" + "=" * 60)
    print("DATASET STATS")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Tier distribution: {df['tier'].value_counts().to_dict()}")
    print(f"Source distribution: {df['source'].value_counts().to_dict()}")
    print(f"Word count: mean={df['word_count'].mean():.1f}, "
          f"min={df['word_count'].min()}, max={df['word_count'].max()}")

    # Specific contamination checks
    for name, pattern in [
        ("SD params (--ar)", r"--ar\s+\d+"),
        ("Screenplay (^Camera:)", r"(?i)^Camera\s*:"),
        ("Leakage (requirements:)", r"(?i)requirements?\s*:"),
        ("Character names", CHARACTER_NAME_PATTERN),
    ]:
        count = df["response"].str.contains(pattern, regex=True, na=False).sum()
        flag = "✅" if count == 0 else "❌"
        print(f"  {flag} {name}: {count}")

    print("\n" + "=" * 60)
    print("SAMPLE OUTPUTS")
    print("=" * 60)
    for source in df["source"].unique():
        sub = df[df["source"] == source]
        if len(sub) > 0:
            row = sub.sample(1, random_state=42).iloc[0]
            print(f"\n--- {source.upper()} ---")
            print(f"Instruction: {row['instruction']}")
            print(f"Response:    {row['response']}")
            print(f"Words: {row['word_count']}, Tier: {row['tier']}")


def stage4_merge_and_push(
    dfs: List[pd.DataFrame],
    target: str,
    dry_run: bool = False,
) -> Optional[DatasetDict]:
    """Final quality filter, dedup, validate, split, and push."""
    print("\n" + "=" * 60)
    print("STAGE 4: Merge + Filter + Push")
    print("=" * 60)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined pre-filter: {len(combined)} rows")

    # Recalculate word_count from response (normalize)
    combined["word_count"] = combined["response"].apply(count_words)
    combined["tier"] = combined["word_count"].apply(assign_tier)

    # Final quality filter
    filtered = final_quality_filter(combined)

    # Fuzzy dedup
    deduped = fuzzy_deduplicate(filtered, threshold=85)

    # Stats and samples
    validate_and_sample(deduped)

    # Post-filter validation (50-row sample; halt if >5% fail)
    should_halt = post_filter_validation(deduped, sample_size=50, fail_threshold=0.05)
    if should_halt and not dry_run:
        print("\n❌ Halting — fix quality issues before pushing.")
        return None

    min_rows = 200
    if len(deduped) < min_rows:
        print(f"\n⚠️  Only {len(deduped)} rows after filtering (minimum {min_rows})")
        if not dry_run:
            print("   Increase stage limits and rerun.")
            return None

    if dry_run:
        print(f"\n[DRY RUN] Would push {len(deduped)} rows to: {target}")
        return None

    # Shuffle and 90/10 split
    deduped = deduped.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(deduped) * 0.9)
    train_df = deduped.iloc[:split_idx]
    val_df = deduped.iloc[split_idx:]
    print(f"\nSplit: {len(train_df)} train, {len(val_df)} validation")

    columns = ["instruction", "response", "tier", "word_count", "text", "source"]
    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df[columns]),
        "validation": Dataset.from_pandas(val_df[columns]),
    })

    print(f"\nPushing to Hub: {target}")
    try:
        dataset_dict.push_to_hub(target, token=HF_TOKEN, private=True)
        print(f"✅ Pushed: https://huggingface.co/datasets/{target}")
    except Exception as e:
        print(f"❌ Push failed: {e}")
        local_path = f"./data/{target.split('/')[-1]}"
        os.makedirs(local_path, exist_ok=True)
        deduped.to_json(f"{local_path}/data.jsonl", orient="records", lines=True)
        print(f"   Saved locally: {local_path}/data.jsonl")
        return None

    return dataset_dict


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build Deforum Prompt Dataset v6 (Decoupled Instruction/Synthesis Sources)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Synthesize 5 samples per stage, print stats, no push")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target", type=str, default="Limbicnation/deforum-prompt-lora-dataset-v6",
                        help="Target dataset ID on Hub")
    parser.add_argument("--ollama-model", type=str, default=OLLAMA_MODEL,
                        help="Ollama model for synthesis (default: qwen3:4b)")
    parser.add_argument("--stage1-limit", type=int, default=len(SCENE_SEEDS),
                        help=f"Max rows from scene seeds (default: {len(SCENE_SEEDS)}, all seeds)")
    parser.add_argument("--stage2-limit", type=int, default=700,
                        help="Max rows from Creative Writing (default: 700)")
    parser.add_argument("--stage3-limit", type=int, default=700,
                        help="Max rows from Gutenberg Sci-Fi (default: 700)")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-stage3", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Print per-row synthesis details")
    args = parser.parse_args()

    model = args.ollama_model

    print("=" * 60)
    print("BUILD DATASET v5 (HARD QUALITY GATES)")
    print("=" * 60)
    print(f"Mode:    {'DRY RUN' if args.dry_run else 'FULL'}")
    print(f"Seed:    {args.seed}")
    print(f"Model:   {model}")
    print(f"Limits:  Stage1={args.stage1_limit}, Stage2={args.stage2_limit}, "
          f"Stage3={args.stage3_limit}")
    print(f"Target:  {args.target}")
    print(f"Seeds:   {len(SCENE_SEEDS)} curated scene seeds")

    # Verify Ollama is running and model is available
    print("\nChecking Ollama...")
    try:
        model_list = ollama_lib.list()
        available = [m.model for m in model_list.models]
        print(f"  Available models: {available}")
        if not any(model in m for m in available):
            print(f"❌ Model '{model}' not found. Available: {available}")
            print(f"   Run: ollama pull {model}")
            sys.exit(1)
        print(f"  ✅ Model '{model}' available")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Run: ollama serve")
        sys.exit(1)

    rng = random.Random(args.seed)
    start_time = time.time()
    dfs = []

    if not args.skip_stage1:
        df1 = stage1_scene_seeds(
            rng, limit=args.stage1_limit, dry_run=args.dry_run, model=model, debug=args.debug
        )
        if len(df1) > 0:
            dfs.append(df1)

    if not args.skip_stage2:
        df2 = stage2_creative_writing(
            rng, limit=args.stage2_limit, dry_run=args.dry_run, model=model, debug=args.debug
        )
        if len(df2) > 0:
            dfs.append(df2)

    if not args.skip_stage3:
        df3 = stage3_gutenberg(
            rng, limit=args.stage3_limit, dry_run=args.dry_run, model=model, debug=args.debug
        )
        if len(df3) > 0:
            dfs.append(df3)

    if not dfs:
        print("\n❌ No data produced from any stage")
        sys.exit(1)

    stage4_merge_and_push(dfs, target=args.target, dry_run=args.dry_run)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"DONE — {elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
