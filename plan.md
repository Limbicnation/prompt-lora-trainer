# v3 Dataset Pipeline + Retrain Plan

> **Status:** Ready for implementation  
> **Last Updated:** 2026-02-17  
> **Decision:** Use LLM synthesis (Ollama + qwen3:4b) for Stages 2 & 3

---

## Context & Problem Statement

The v3 LoRA adapter (trained on v2 dataset) exhibits critical quality issues:

| Issue | Prevalence | Root Cause |
|-------|------------|------------|
| "Sarah" character leakage | 51.9% (2,522/4,860 rows) | Over-representation in source |
| "chiaroscuro" overuse | 50.0% (2,431/4,860 rows) | Template slotting repetition |
| "comes into view" boilerplate | 7.4% (359/4,860 rows) | Rigid template structure |

**Dataset format issue:** v2 uses custom `### Instruction: / ### Response:` format (no `text` column), not Qwen3 chat template.

### Repository State

| File | Status | Action Needed |
|------|--------|---------------|
| `scripts/train_sft.py` | Modified (trackio support) | ☐ Commit |
| `configs/sft_qwen3_4b_deforum_v3.yaml` | Exists | ☐ Update dataset_id |
| `Modelfile.deforum-v3` | Untracked | ☐ Commit |

---

## Implementation Checklist

### ☐ Step 0: Commit Current Changes

```bash
git add scripts/train_sft.py configs/sft_qwen3_4b_deforum_v3.yaml Modelfile.deforum-v3
git commit -m "feat: trackio support and v3 config setup"
```

---

### ☐ Step 1: Create `scripts/build_dataset_v3.py`

**Approach:** Single PEP 723 script with LLM synthesis for Stages 2 & 3.

#### Script Structure

```python
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
#   "ollama>=0.1.0",  # For LLM synthesis
# ]
# ///
```

#### Stage 1: Clean v2 Dataset (~4,400 rows)

| Task | Implementation Details |
|------|------------------------|
| Load | DuckDB `hf://datasets/Limbicnation/deforum-prompt-lora-dataset-v2@~parquet/default/train/*.parquet` |
| Diversify "Sarah" | Regex replace with pool: "the figure", "the silhouette", "Elena", "Mara", "Yuki", "the woman", "the stranger", "the protagonist", "a shadow", "someone" |
| Diversify "chiaroscuro" | Replace 80% with: rim lighting, neon glow, volumetric haze, backlit silhouette, split lighting, candlelight, diffused overhead, practical lighting |
| Clean | Drop rows where `instruction` or `response` has <3 words |
| Strip meta | Remove: "We respect the original creators", "Your descriptive prowess", "comes into view via", "descriptive prowess", "Scene \d+:" |
| Format | Qwen3 chat template (see below) |
| Tag | `source="v2_cleaned"` |

**Qwen3 Chat Template (v2_cleaned - WITH Technical Params):**
```python
text = f"""<|im_start|>system
You are a cinematic video prompt generator specializing in the De Forum Art Film aesthetic.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}

Technical Parameters:
- Negative Prompt: {negative}
- Tags: {tags_str}
- Settings: --ar {aspect_ratio} --model {model} --seed {seed}<|im_end|>"""
```

#### Stage 2: Creative_Writing-ShareGPT (~800-1,200 rows)

| Step | Action |
|------|--------|
| 1 | Load via DuckDB, extract first human+gpt exchange |
| 2 | Score for visual/atmospheric density (keyword lists below) |
| 3 | Filter top 20% (~1,300 rows) |
| 4 | **LLM Synthesis** via Ollama (`qwen3:4b`) |
| 5 | Assign tiers by word count |
| 6 | Format with simplified chat template, tag `source="creative_writing"` |

**Scoring Keywords:**
- **Visual:** light, shadow, glow, shimmer, dark, bright, haze, fog, smoke, dust, flame, neon, silhouette, reflection
- **Atmospheric:** silence, whisper, echo, wind, rain, thunder, creak, hum, pulse, breathe
- **Texture:** grain, rough, smooth, cold, warm, damp, velvet, rust, glass, metal
- **Movement:** drift, float, crawl, sweep, cascade, ripple, flicker, sway, surge

#### Stage 3: Gutenberg Sci-Fi (~500-700 rows)

| Step | Action |
|------|--------|
| 1 | Load ~500 books via DuckDB |
| 2 | Strip headers/footers with regex `\*\*\* START OF.*?\*\*\*` and `\*\*\* END OF.*?\*\*\*` |
| 3 | Chunk into 200-500 word segments at `\n\n` |
| 4 | Score with Stage 2 keywords + sci-fi visuals |
| 5 | Filter top ~700 chunks |
| 6 | **LLM Synthesis** via Ollama |
| 7 | Format, assign tiers, tag `source="gutenberg_scifi"` |

**Additional Sci-Fi Keywords:** hologram, viewport, console, starfield, nebula, reactor, dome, corridor, airlock, hull

---

### LLM Synthesis Implementation

**Pre-requisite:**
```bash
ollama pull qwen3:4b
```

**Synthesis Function:**
```python
import ollama
from tqdm import tqdm

def synthesize_cinematic_prompt(source_text: str, instruction_hint: str = "") -> tuple:
    """
    Rewrite source text as De Forum cinematic prompt.
    Returns: (instruction, response)
    """
    system_prompt = """You are a cinematic video prompt engineer specializing in the De Forum aesthetic: noir-influenced, minimalist, psychologically intense art film style."""
    
    user_prompt = f"""Source text:
---
{source_text[:800]}  # Truncate to avoid context limits
---

Rewrite this as a cinematic video diffusion prompt. Requirements:
- 40-80 words
- Include: subject, camera movement, lighting description, mood
- NO character names (use "the figure", "a silhouette", "the subject")
- NO repetitive phrases like "comes into view"
- Varied lighting (avoid defaulting to "chiaroscuro")
- Film grain, atmospheric, contemplative tone

Output ONLY a JSON object:
{{"instruction": "brief user request", "response": "the cinematic prompt"}}"""

    response = ollama.chat(
        model='qwen3:4b',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        options={
            'temperature': 0.7,
            'top_p': 0.8,
            'num_predict': 200,
            'repeat_penalty': 1.3
        }
    )
    
    # Parse JSON response
    import json
    try:
        result = json.loads(response['message']['content'])
        return result['instruction'], result['response']
    except:
        # Fallback parsing
        return instruction_hint, response['message']['content'][:300]

# Batch processing with progress bar
def batch_synthesize(texts: list, batch_size: int = 10) -> list:
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Synthesizing"):
        batch = texts[i:i+batch_size]
        for text in batch:
            results.append(synthesize_cinematic_prompt(text))
    return results
```

**Simplified Chat Template (Stages 2 & 3 - NO Technical Params):**
```python
text = f"""<|im_start|>system
You are a cinematic video prompt generator specializing in the De Forum Art Film aesthetic.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""
```

**Estimated Time:** ~30-60 min for ~2,000 rows on RTX 4090

---

#### Stage 4: Merge, Deduplicate, Upload

| Task | Details |
|------|---------|
| Combine | All 3 DataFrames with standardized columns |
| Columns | `instruction`, `response`, `tier`, `word_count`, `text`, `source` |
| Deduplicate | `thefuzz` token_sort_ratio >= 85, blocked by tier + hash |
| Validate | Non-empty fields, chat tokens present, word counts in bounds |
| Targets | Sarah < 10%, chiaroscuro < 15% |
| Split | 90/10 train/val, seed=42 |
| Push | `Limbicnation/deforum-prompt-lora-dataset-v3` |

**Tier Word Counts:**
- short: 15-30 words
- medium: 30-60 words  
- detailed: 60-100 words

---

### ☐ Step 2: Run Dry-Run + Inspect

```bash
# Ensure ollama is running and model is pulled
ollama pull qwen3:4b

# Run dry-run
conda run -n prompt-lora-trainer uv run scripts/build_dataset_v3.py --dry-run --seed 42
```

**Verify:**
- [ ] Sarah < 10%
- [ ] chiaroscuro < 15%
- [ ] No meta-commentary
- [ ] All rows have chat template tokens (`<|im_start|>`, `<|im_end|>`)
- [ ] Tier distribution ~33/34/33
- [ ] Source breakdown reasonable
- [ ] 5+ manual samples per source look good

---

### ☐ Step 3: Push v3 Dataset

```bash
conda run -n prompt-lora-trainer uv run scripts/build_dataset_v3.py \
  --target Limbicnation/deforum-prompt-lora-dataset-v3 \
  --seed 42
```

---

### ☐ Step 4: Update Config + Retrain

**4.1 Update config:**
```yaml
# configs/sft_qwen3_4b_deforum_v3.yaml
dataset_id: "Limbicnation/deforum-prompt-lora-dataset-v3"
```

**4.2 Dry-run:**
```bash
conda run -n prompt-lora-trainer python scripts/train_sft.py \
  --config configs/sft_qwen3_4b_deforum_v3.yaml --dry-run
```

**4.3 Full training:**
```bash
# IMPORTANT: Use conda env (torch 2.6.0+cu124), NOT .venv-train
conda run -n prompt-lora-trainer python scripts/train_sft.py \
  --config configs/sft_qwen3_4b_deforum_v3.yaml
```

**Training Config:**
- QLoRA NF4, bf16 compute
- r=16, alpha=32, attention-only
- eval every 25 steps, early stopping patience=3
- save_steps=50 (must be >= eval_steps)

---

### ☐ Step 5: GGUF Export + Ollama Deploy

Update `convert_and_upload.sh` paths for v3 output, then run:
```bash
./convert_and_upload.sh
```

---

## CLI Flags for build_dataset_v3.py

| Flag | Description | Default |
|------|-------------|---------|
| `--dry-run` | Run all stages, print stats, don't push | False |
| `--skip-creative` | Skip Stage 2 | False |
| `--skip-gutenberg` | Skip Stage 3 | False |
| `--synthesize-with-ollama` | Enable LLM synthesis | True |
| `--ollama-model` | Model for synthesis | `qwen3:4b` |
| `--batch-size` | Ollama batch size | 10 |
| `--seed` | Random seed | 42 |
| `--target` | Target dataset ID | `Limbicnation/deforum-prompt-lora-dataset-v3` |

---

## Code References

| Pattern | File | Lines |
|---------|------|-------|
| DuckDB loading | `reformat_dataset_v2.py` | 605-623 |
| Qwen3 chat template | `upload_dataset_to_hub.py` | 46-80 |
| Camera inference | `process_deforum_data.py` | 288-304 |
| Visual extraction | `process_deforum_data.py` | 151-172 |
| Tier templates | `reformat_dataset_v2.py` | 44-131 |
| Tier distribution | `reformat_dataset_v2.py` | 30-37 |
| Response parsing | `reformat_dataset_v2.py` | 155-260 |

---

## Success Criteria

- [ ] Dataset pushed to Hub with train/val splits
- [ ] Sarah mentions < 10% of rows
- [ ] chiaroscuro mentions < 15% of rows
- [ ] No "comes into view" boilerplate
- [ ] All rows properly Qwen3 chat formatted
- [ ] Training completes with improving eval_loss
- [ ] Early stopping triggers before epoch 3
- [ ] Ollama test shows diverse lighting, no character leakage
