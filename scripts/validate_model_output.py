#!/usr/bin/env python3
"""Model-output validation for the dual-stream video-prompt LoRA.

Generates prompts FROM the fine-tuned model (and the base model as a pre-fine-tuned
baseline) on a held-out concept set, then scores them against the gates declared in
`validation-plan.mf`: Action Verb Density (AVD), Spatial-Temporal Consistency (STC),
Formatting Compliance (FC), plus corpus guardrails (hallucination, grain, diversity).

The v8 model was trained on the raw `### Instruction/### Response` text format
(scripts/train_sft.py:130), NOT the Qwen3 chat template — this script reproduces that
format exactly. Using a chat template here silently invalidates every metric.

Usage:
    python scripts/validate_model_output.py --smoke        # fast format/parse self-check
    python scripts/validate_model_output.py                # full run -> generations + metrics
    python scripts/validate_model_output.py --no-base      # skip baseline (faster)

Exit code is non-zero if any HARD gate fails (for CI/CD gating).
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = Path(__file__).resolve().parent.parent
ADAPTER_DIR = REPO / "outputs" / "qwen3-4b-deforum-video-dual-stream-lora-v1"
BASE_ID = "Qwen/Qwen3-4B-Instruct-2507"
TRAIN_DATA = REPO / "data" / "dual_stream_video_v8.jsonl"
TEST_CONCEPTS = REPO / "data" / "validation_test_concepts.jsonl"
GEN_OUT = REPO / "data" / "validation_generations_v1.jsonl"
METRICS_OUT = REPO / "reports" / "validation_metrics_v1.json"

# Exact dialect prefixes the model was trained on (derived from the v8 instruction field).
DIALECT_PREFIXES = {
    "wan_video": "Generate a WanVideo (UMT5) prompt for:",
    "ltx_video": "Generate an LTX-Video prompt for:",
    "compact_caption": "Write a compact video caption for:",
}

# --- lexicons -------------------------------------------------------------------------
# AVD numerator: dynamic motion verbs + camera-move tokens (mirrors validation-plan.mf).
MOTION = {
    "dolly",
    "pan",
    "tilt",
    "track",
    "tracking",
    "push",
    "pull",
    "zoom",
    "orbit",
    "crane",
    "glide",
    "drift",
    "sweep",
    "rise",
    "rises",
    "descend",
    "descends",
    "rotate",
    "rotates",
    "handheld",
    "aerial",
    "pans",
    "tilts",
    "tracks",
    "pushes",
    "pulls",
    "zooms",
    "glides",
    "sweeps",
    "drifts",
    "moves",
    "moving",
    "races",
    "rushing",
    "soaring",
    "circling",
}
# Broader camera vocabulary for FC structural checks (presence + lead-in).
CAMERA = MOTION | {
    "camera",
    "shot",
    "angle",
    "close-up",
    "closeup",
    "wide",
    "macro",
    "static",
    "steadicam",
    "pov",
    "low-angle",
    "high-angle",
    "overhead",
    "pullback",
    "push-in",
    "slow",
}
CAMERA_LEAD_PHRASES = (
    "slow ",
    "a slow",
    "the camera",
    "camera ",
    "wide ",
    "close-up",
    "close up",
    "aerial",
    "tracking",
    "handheld",
    "static ",
    "low-angle",
    "high-angle",
    "overhead",
    "macro",
    "push-in",
    "pull",
    "dolly",
    "pan ",
    "tilt",
    "crane",
    "drone",
    "orbit",
)
# Hallucinated entities (concept-ungrounded) — from the v8 dataset audit.
HALLUCINATION = {"zombie", "alien", "mars", "bloodstain", "diamond-illuminated", "refugee", "ghost"}
STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "to",
    "and",
    "with",
    "her",
    "his",
    "their",
    "as",
    "by",
    "for",
    "from",
    "into",
    "over",
    "under",
    "behind",
    "across",
    "s",
}
LEAK_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
CLI_FLAG_RE = re.compile(r"--(ar|model|seed|stylize|niji|chaos|v|q|no)\b", re.IGNORECASE)
PREAMBLE = ("###", "- ", "* ", "1.", "```", "sure", "here", "okay", "certainly", "as an", "i ")

# Per-dialect word-count windows + gate thresholds (kept in sync with validation-plan.mf).
WINDOW = {"wan_video": (21, 56), "ltx_video": (20, 50), "compact_caption": (5, 20)}
# stc_min calibrated to the sbert-MiniLM grounding distribution (clean prompt ~0.72,
# hallucinated ~0.42, non-camera-led ~0.62): the gate separates on those deductions.
THRESHOLDS = {
    "wan_video": {"avd_min": 0.10, "stc_min": 0.70, "fc_min": 0.90, "camera_lead_min": 0.85},
    "ltx_video": {"avd_min": 0.08, "stc_min": 0.70, "fc_min": 0.90, "camera_lead_min": 0.85},
    "compact_caption": {"avd_min": 0.00, "stc_min": 0.68, "fc_min": 0.95, "camera_lead_min": None},
}
GUARDRAILS = {
    "hallucination_rate_max": 0.01,
    "film_grain_freq_max": 0.15,
    "distinct_2_min": 0.60,
    "single_camera_move_share_max": 0.35,
}
REGRESSION = {"avd_delta_min": 0.03, "fc_delta_min": 0.10}


def words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z'\-]*", text.lower())


_SBERT = None


def grounding_score(concept: str, text: str) -> float:
    """Concept-grounding in [0,1]. sbert_cosine (per validation-plan.mf) with a lexical
    fallback so the scorer still runs if sentence-transformers is unavailable."""
    global _SBERT
    if _SBERT is None:
        try:
            from sentence_transformers import SentenceTransformer, util

            _SBERT = (SentenceTransformer("all-MiniLM-L6-v2"), util)
        except Exception:
            _SBERT = False
    if _SBERT:
        model, util = _SBERT
        emb = model.encode([concept, text], convert_to_tensor=True, normalize_embeddings=True)
        return max(0.0, float(util.cos_sim(emb[0], emb[1])))
    # ponytail: lexical fallback only — token overlap under-measures paraphrase.
    content = {w for w in words(concept) if w not in STOPWORDS and len(w) > 2}
    return min(1.0, len(set(words(text)) & content) / max(1, len(content)))


def leads_with_camera(text: str) -> bool:
    low = text.lower().lstrip()
    if low.startswith(CAMERA_LEAD_PHRASES):
        return True
    return any(w in CAMERA for w in words(text)[:3])


def score_prompt(text: str, concept: str, dialect: str) -> dict:
    toks = words(text)
    n = max(1, len(toks))
    tokset = set(toks)

    avd = sum(1 for w in toks if w in MOTION) / n

    # Formatting Compliance — binary checks. Camera-direction checks apply only to the
    # video dialects; compact_caption is descriptive prose and is scored on the rest.
    lo, hi = WINDOW[dialect]
    camera_lead = leads_with_camera(text)
    fc_checks = {
        "wordcount_in_window": lo <= len(toks) <= hi,
        "no_cli_flags": CLI_FLAG_RE.search(text) is None,
        "no_chat_token_leak": not any(t in text for t in LEAK_TOKENS),
        "no_markdown_or_preamble": not text.lower().lstrip().startswith(PREAMBLE),
    }
    if dialect != "compact_caption":
        fc_checks["camera_token_present"] = any(w in CAMERA for w in toks)
        fc_checks["leads_with_camera_verb"] = camera_lead
    fc = sum(fc_checks.values()) / len(fc_checks)

    # Spatial-Temporal Consistency.
    grounding = grounding_score(concept, text)
    halluc = sorted(e for e in HALLUCINATION if e in tokset and e not in concept.lower())
    single_scene = 0.0 if halluc else 1.0
    temporal = 1.0 if camera_lead else (0.7 if dialect == "compact_caption" else 0.5)
    stc = 0.5 * grounding + 0.3 * single_scene + 0.2 * temporal

    return {
        "word_count": len(toks),
        "avd": round(avd, 4),
        "fc": round(fc, 4),
        "stc": round(stc, 4),
        "camera_lead": camera_lead,
        "grain": "grain" in tokset,
        "hallucinated": halluc,
        "fc_checks": fc_checks,
    }


def aggregate(records: list[dict]) -> dict:
    """Per-dialect aggregates over one model's generations."""
    out = {}
    by_dialect: dict[str, list[dict]] = {}
    for r in records:
        by_dialect.setdefault(r["dialect"], []).append(r)
    for dialect, recs in by_dialect.items():
        n = len(recs)
        scores = [r["score"] for r in recs]
        bigrams = Counter()
        cam_counts = Counter()
        for r in recs:
            w = words(r["output"])
            bigrams.update(zip(w, w[1:]))
            for cam in (
                "dolly",
                "pan",
                "tilt",
                "track",
                "push",
                "zoom",
                "crane",
                "orbit",
                "static",
                "handheld",
            ):
                if cam in w:
                    cam_counts[cam] += 1
        distinct2 = (len(bigrams) / sum(bigrams.values())) if bigrams else 0.0
        max_cam_share = (max(cam_counts.values()) / n) if cam_counts else 0.0
        out[dialect] = {
            "n": n,
            "avd": round(sum(s["avd"] for s in scores) / n, 4),
            "fc": round(sum(s["fc"] for s in scores) / n, 4),
            "stc": round(sum(s["stc"] for s in scores) / n, 4),
            "camera_lead_rate": round(sum(s["camera_lead"] for s in scores) / n, 4),
            "grain_rate": round(sum(s["grain"] for s in scores) / n, 4),
            "hallucination_rate": round(sum(bool(s["hallucinated"]) for s in scores) / n, 4),
            "median_wc": sorted(s["word_count"] for s in scores)[n // 2],
            "distinct_2": round(distinct2, 4),
            "max_single_camera_share": round(max_cam_share, 4),
        }
    return out


def check_gates(adapter_agg: dict, base_agg: dict | None) -> list[tuple]:
    """Return [(gate, severity, passed, detail)]. severity in {hard, soft}."""
    rows = []
    for dialect, th in THRESHOLDS.items():
        a = adapter_agg.get(dialect)
        if not a:
            continue
        rows.append(
            (f"{dialect}.fc>={th['fc_min']}", "hard", a["fc"] >= th["fc_min"], f"{a['fc']:.3f}")
        )
        rows.append(
            (
                f"{dialect}.stc>={th['stc_min']}",
                "hard",
                a["stc"] >= th["stc_min"],
                f"{a['stc']:.3f}",
            )
        )
        rows.append(
            (
                f"{dialect}.avd>={th['avd_min']}",
                "soft",
                a["avd"] >= th["avd_min"],
                f"{a['avd']:.3f}",
            )
        )
        if th["camera_lead_min"] is not None:
            rows.append(
                (
                    f"{dialect}.camera_lead>={th['camera_lead_min']}",
                    "soft",
                    a["camera_lead_rate"] >= th["camera_lead_min"],
                    f"{a['camera_lead_rate']:.3f}",
                )
            )
        rows.append(
            (
                f"{dialect}.hallucination<={GUARDRAILS['hallucination_rate_max']}",
                "hard",
                a["hallucination_rate"] <= GUARDRAILS["hallucination_rate_max"],
                f"{a['hallucination_rate']:.3f}",
            )
        )
        rows.append(
            (
                f"{dialect}.grain<={GUARDRAILS['film_grain_freq_max']}",
                "soft",
                a["grain_rate"] <= GUARDRAILS["film_grain_freq_max"],
                f"{a['grain_rate']:.3f}",
            )
        )
        rows.append(
            (
                f"{dialect}.distinct2>={GUARDRAILS['distinct_2_min']}",
                "soft",
                a["distinct_2"] >= GUARDRAILS["distinct_2_min"],
                f"{a['distinct_2']:.3f}",
            )
        )
        rows.append(
            (
                f"{dialect}.single_cam<={GUARDRAILS['single_camera_move_share_max']}",
                "soft",
                a["max_single_camera_share"] <= GUARDRAILS["single_camera_move_share_max"],
                f"{a['max_single_camera_share']:.3f}",
            )
        )
        if base_agg and dialect in base_agg:
            delta = a["fc"] - base_agg[dialect]["fc"]
            rows.append(
                (
                    f"{dialect}.fc_delta_vs_base>={REGRESSION['fc_delta_min']}",
                    "hard",
                    delta >= REGRESSION["fc_delta_min"],
                    f"{delta:+.3f}",
                )
            )
    return rows


# --- model -----------------------------------------------------------------------------
def load(adapter_dir: Path = ADAPTER_DIR):
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at {adapter_dir}. Train it first, download it from the Hub "
            f"(Limbicnation/qwen3-4b-deforum-video-dual-stream-lora-v1), or pass --adapter-dir."
        )
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"📥 Loading base: {BASE_ID} (4bit)")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, device_map="auto", quantization_config=bnb, trust_remote_code=True
    )
    print(f"📥 Loading adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    tok = AutoTokenizer.from_pretrained(str(adapter_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


@torch.inference_mode()
def generate(model, tok, dialect: str, concept: str, max_new_tokens: int) -> str:
    instruction = f"{DIALECT_PREFIXES[dialect]} {concept}"
    # Exact training format — NOT a chat template.
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy -> reproducible
        repetition_penalty=1.2,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0][enc.input_ids.shape[1] :], skip_special_tokens=True).strip()
    # Cut any spillover into a new section header.
    return re.split(r"\n#{1,3}\s|\n### ", text)[0].strip()


def run(model, tok, concepts: list[dict], with_base: bool, max_new_tokens: int) -> list[dict]:
    records = []
    variants = [("adapter", False)] + ([("base", True)] if with_base else [])
    total = len(concepts) * len(DIALECT_PREFIXES) * len(variants)
    i = 0
    for c in concepts:
        for dialect in DIALECT_PREFIXES:
            for variant, disable in variants:
                i += 1
                if disable:
                    with model.disable_adapter():
                        out = generate(model, tok, dialect, c["concept"], max_new_tokens)
                else:
                    out = generate(model, tok, dialect, c["concept"], max_new_tokens)
                rec = {
                    "variant": variant,
                    "dialect": dialect,
                    "split": c.get("split", "ood"),
                    "concept": c["concept"],
                    "output": out,
                    "score": score_prompt(out, c["concept"], dialect),
                }
                records.append(rec)
                print(
                    f"[{i}/{total}] {variant:7s} {dialect:15s} "
                    f"fc={rec['score']['fc']:.2f} stc={rec['score']['stc']:.2f} "
                    f"avd={rec['score']['avd']:.2f} | {out[:60]}"
                )
    return records


def smoke(model, tok) -> int:
    """Smallest check that fails if format/parse breaks."""
    concept = "a lighthouse on a storm-battered cliff at dawn"
    out = generate(model, tok, "wan_video", concept, 80)
    assert out, "empty generation"
    assert not any(t in out for t in LEAK_TOKENS), f"chat-token leak: {out!r}"
    assert "### " not in out, f"section header leaked into output: {out!r}"
    with model.disable_adapter():
        base_out = generate(model, tok, "wan_video", concept, 80)
    assert base_out, "empty base generation"
    print("\n✅ smoke: adapter + base generate clean, parseable prompts")
    print(f"   adapter: {out}")
    print(f"   base   : {base_out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="fast 1-concept format/parse self-check")
    p.add_argument("--no-base", action="store_true", help="skip base-model baseline generation")
    p.add_argument("--limit", type=int, default=0, help="limit number of concepts (0 = all)")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--adapter-dir", type=Path, default=ADAPTER_DIR, help="LoRA adapter directory")
    p.add_argument(
        "--rescore",
        action="store_true",
        help="recompute metrics from saved generations (no model load)",
    )
    args = p.parse_args()

    if args.rescore:
        if not GEN_OUT.exists():
            raise FileNotFoundError(f"No saved generations at {GEN_OUT}; run a full pass first.")
        records = [json.loads(line) for line in GEN_OUT.read_text().splitlines() if line.strip()]
        for r in records:  # re-score stored outputs with the current metric definitions
            r["score"] = score_prompt(r["output"], r["concept"], r["dialect"])
        GEN_OUT.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return report(records)

    model, tok = load(args.adapter_dir)
    if args.smoke:
        return smoke(model, tok)

    concepts = [json.loads(line) for line in TEST_CONCEPTS.read_text().splitlines() if line.strip()]
    # Leak-check OOD concepts against the training set. The training jsonl is gitignored, so on a
    # fresh clone it may be absent — skip the check (with a warning) rather than crash.
    if TRAIN_DATA.exists():
        train_concepts = {
            json.loads(line)["original_concept"].lower()
            for line in TRAIN_DATA.read_text().splitlines()
            if line.strip()
        }
        leaked = [
            c["concept"]
            for c in concepts
            if c.get("split") == "ood" and c["concept"].lower() in train_concepts
        ]
        assert not leaked, f"OOD concepts leaked into training data: {leaked}"
    else:
        print(f"⚠️  {TRAIN_DATA} not found — skipping OOD leak-check.")
    if args.limit:
        concepts = concepts[: args.limit]

    records = run(
        model, tok, concepts, with_base=not args.no_base, max_new_tokens=args.max_new_tokens
    )
    GEN_OUT.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return report(records)


def report(records: list[dict]) -> int:
    """Aggregate, gate, persist metrics, print results. Returns 1 if any hard gate fails."""
    adapter_agg = aggregate([r for r in records if r["variant"] == "adapter"])
    base_recs = [r for r in records if r["variant"] == "base"]
    base_agg = aggregate(base_recs) if base_recs else None

    gates = check_gates(adapter_agg, base_agg)
    METRICS_OUT.parent.mkdir(exist_ok=True)
    METRICS_OUT.write_text(
        json.dumps(
            {
                "adapter": adapter_agg,
                "base": base_agg,
                "gates": [
                    {"gate": g, "severity": s, "passed": ok, "value": v} for g, s, ok, v in gates
                ],
            },
            indent=2,
        )
    )

    print("\n" + "=" * 78 + "\nGATE RESULTS\n" + "=" * 78)
    hard_fail = 0
    for gate, severity, passed, detail in gates:
        mark = "✅" if passed else ("❌" if severity == "hard" else "⚠️ ")
        hard_fail += int(severity == "hard" and not passed)
        print(f"{mark} [{severity:4s}] {gate:48s} = {detail}")
    print(f"\n📄 generations → {GEN_OUT}\n📊 metrics → {METRICS_OUT}")
    print(
        f"\n{'❌ HARD GATE FAILURES: ' + str(hard_fail) if hard_fail else '✅ all hard gates passed'}"
    )
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
