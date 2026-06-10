# Dataset Upgrade Plan: deforum-v7 → Multi-Dialect Video Prompt Schema

This document plans the upgrade of `Limbicnation/deforum-prompt-lora-dataset-v7` into a
multi-dialect dataset for latent video diffusion models (LTX-Video, WanVideo/Wan2.1). It was
revised after auditing the **actual** dataset against the repo's existing tooling (see the audit
verdict in §6).

---

## 1. Rationale (corrected)

The v7 corpus is **not** legacy single-field Deforum keyframe strings. Measured schema
(1,547 train / 172 val rows): `instruction`, `response`, `tier`, `word_count`, `text`, `source` —
an instruction→response SFT set whose `response` is unified cinematic prose. There is no
frame-weighted (`0:(...)`) syntax anywhere.

The real conditioning problems for video models are:

1. **Non-visual contamination — 47.1% of responses** contain tokens a camera cannot capture
   (scent/sound/silence/memory/grief). These degrade text-to-video alignment.
2. **No model-dialect separation.** A single prose field is not routed for LTX-Video (T5-XXL,
   motion-centric shot lists) vs WanVideo (UMT5, Subject + Scene + Motion).

What is already healthy and must be preserved: camera-movement directives (89.2% of rows; 72.9%
*start* with a camera clause) and lighting/atmosphere vocabulary (99.0%).

---

## 2. Target Schema (flat, inference-routed — not nested)

Routing happens **at inference via the instruction prefix**, fanned out to flat
one-dialect-per-row records. This matches the repo's existing, deliberate decision (see
`scripts/generate_dual_stream_dataset.py`) and avoids training the LoRA to emit nested dicts.

```python
# One row PER (concept, target_model) pair
{
    "original_concept": str,   # concept extracted from the source instruction
    "target_model":     str,   # "wan_video" | "ltx_video" | "compact_caption"
    "instruction":      str,   # e.g. "Generate a WanVideo (UMT5) prompt for: <concept>"
    "response":         str,   # the prompt in that dialect (non-visual content stripped)
    # optional: "text"         # Qwen chat-template render for train_sft.py (--render-text)
}
```

Dialect budgets **calibrated to the 48w-median source** (and validated on a 10-row Ollama smoke
run, see §5):

| target_model | shape | word band (gate) | hard gate |
| :--- | :--- | :--- | :--- |
| `wan_video` | Subject + Scene + Motion prose | 20–80w | ≥1 camera term, ≥1 lighting term, **0 non-visual tokens** |
| `ltx_video` | motion-centric shot prose | 20–80w | ≥1 camera term, **0 non-visual tokens** |
| `compact_caption` | dense global caption | ≤25w | **0 non-visual tokens**, no buzzwords |

> **Why no `detailed_scene_description` (40–100w) or separate `temporal_flow` (30–70w)?**
> The source median is 48 words (max 75; **0/1547 rows ≥90w**). Asking one 48-word row to yield a
> 25w caption + 40–100w scene + 30–70w motion forces the LLM to *invent* ~100–200 words of content,
> causing semantic drift. After stripping the ~47% non-visual padding, the visual core lands near
> 20–30w — so the dialects above are **extractive**, not generative. Camera direction is kept inline
> (already present in 89.2% of rows), not split into a synthesized field.

---

## 3. Implementation: reuse the existing pipeline (do NOT write a new script)

`scripts/generate_dual_stream_dataset.py` already provides TRANSFORM mode, single-LLM fan-out,
per-dialect quality gates, bad-pattern/buzzword rejection, `--dry-run`, Gemini + Ollama backends,
`--render-text`, and `--hub-id` push. The upgrade is delivered by a new **`--dialect-set video`**
switch added to that script:

- `VIDEO_DIALECTS` = `wan_video`, `ltx_video`, `compact_caption` (+ instruction prefixes).
- `VIDEO_DIALECT_GATES` with the bands in §2, cinematic `VIDEO_CAMERA_TERMS` /
  `VIDEO_LIGHTING_TERMS` (motion + atmosphere vocab, not lens specs), and a shared
  **non-visual hard filter** (`NONVISUAL_TERMS` / `_has_nonvisual`).
- `SYSTEM_TRANSFORM_VIDEO` instructs the LLM to rewrite the source into the three dialects and
  **strip all non-visual content**.
- `_extract_style` extended to parse the deforum instruction form (`"...for: <concept>"`).
- Routing/format selected via `SYSTEM_BY` / `SCHEMA_BY` / `DIALECTS_BY` / `GATES_BY` registries.

TRANSFORM mode auto-activates because the dataset has a `response` column. No new script file.

---

## 4. Quality Gates (validated)

Applied per-dialect in `fan_out_and_gate`; a failed dialect drops only that row, not the bundle.

1. **Non-visual rejection (new):** any `scent|sound|silence|memory|grief|...` token → reject.
   Drove output contamination from 47.1% → **0%** on the smoke run.
2. **Word-count compliance:** wan/ltx 20–80w; caption ≤25w.
3. **Camera coverage:** wan + ltx require ≥1 camera term (hard; 89.2% baseline pass).
4. **Lighting coverage:** wan requires ≥1 lighting term (hard; 99.0% baseline pass).
5. **Format / preamble rejection:** reuses existing `BAD_PATTERNS` (markdown, CLI flags,
   "Here is…"); buzzword reject retained (near-noop at 0.1% but free).

---

## 5. Verification & Roadmap

```bash
conda activate prompt-lora-trainer          # NOT .venv-train (CUBLAS issues)

# 1. Dry-run smoke test (no write, no push) — DONE, results below
uv run scripts/generate_dual_stream_dataset.py \
  --dataset Limbicnation/deforum-prompt-lora-dataset-v7 \
  --dialect-set video --backend ollama --ollama-model qwen3:8b \
  --max-rows 10 --dry-run

# 2. Full run + render text + push (private) — pending decision on backend/scale
uv run scripts/generate_dual_stream_dataset.py \
  --dataset Limbicnation/deforum-prompt-lora-dataset-v7 \
  --dialect-set video --render-text \
  --hub-id Limbicnation/deforum-dual-stream-video-v8

# 3. Validate the pushed schema
python scripts/validate_dataset.py --dataset Limbicnation/deforum-dual-stream-video-v8
```

**Smoke-run result (10 concepts, qwen3:8b):** 25 flat rows
(`compact_caption=10, wan_video=9, ltx_video=6`); **0 rows with non-visual tokens**;
**0 captions >25w**; camera + lighting terms retained in all wan/ltx outputs.

Roadmap: (1) ✅ extend pipeline with `--dialect-set video`; (2) ✅ smoke-test on Ollama;
(3) full run + push to `Limbicnation/deforum-dual-stream-video-v8`; (4) add a
`configs/sft_*_deforum_video_dual_stream.yaml` training config (separate task — not covered here).

---

## 6. Audit verdict (why this plan was revised)

The original draft of this plan was rejected on four grounds, all confirmed against the cached
dataset:

| Original claim | Finding |
| :--- | :--- |
| "legacy single-field Deforum keyframe strings" | False — 6-field SFT schema, unified prose, no keyframe syntax |
| `detailed_scene_description` 40–100w + `temporal_flow` 30–70w + caption | Impossible — source median 48w, 0% ≥90w → forces hallucination |
| `model_routed_prompt: dict` (nested) | Regresses the repo's flat-row, inference-time-routing decision |
| build new `upgrade_to_dual_stream_v8.py` | Reinvents the existing dual-stream pipeline |

The one original idea retained and validated: the **non-visual keyword filter**, which targets a
measured 47.1% contamination and now drives it to 0%.
