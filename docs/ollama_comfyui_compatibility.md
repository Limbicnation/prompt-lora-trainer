# Deploying LoRAs to ComfyUI via Ollama

End-to-end reference for taking a freshly trained prompt-generator LoRA (Qwen2.5-7B image,
Qwen3-4B video) into ComfyUI through Ollama: the deployment architecture, setup rules, the node
↔ template routing rules, and the failure modes.

The repo's own integration node is
[`ComfyUI-PromptGenerator`](https://github.com/Limbicnation/ComfyUI-PromptGenerator) (Ollama
backend). Generic Ollama node packs such as [`stavsap/comfyui-ollama`](https://github.com/stavsap/comfyui-ollama)
follow the same `/api/generate` vs `/api/chat` split, so the rules transfer.

---

## TL;DR / Verdict

✅ **`image-v1`/`image-v2` and `dual-stream` are ComfyUI-native.** Their Modelfile templates are
built around `.Prompt`, so they render correctly under **both** Ollama API endpoints — the repo's
`ComfyUI-PromptGenerator` node and any generic Ollama Generate/Chat node.

⚠️ `deforum-v7` (video) uses a `.Messages` template and **only** works through an `/api/chat`-style
node; on an `/api/generate` node it silently drops your prompt. The image models do **not** have this
problem — see [Node ↔ template routing](#node--template-routing) so you don't reintroduce it.

---

## 1. Architecture: compile-and-deploy (not runtime adapters)

Ollama does not load raw Hugging Face `.safetensors` LoRA adapters dynamically via its API. It has a
Modelfile `ADAPTER` directive for *GGUF-format* adapters, but that path is sensitive to quantization
and architecture-shape mismatches. This repo uses the production-stable **compile-and-deploy**
strategy: merge weights statically **before** GGUF conversion. This removes all runtime adapter
overhead, sidesteps quantization issues, and guarantees compatibility with any node calling the
Ollama API.

```
 Base Instruct model  +  LoRA .safetensors  ──merge(bf16)──▶  Merged FP16/BF16 PyTorch
                                                                        │ llama.cpp
                                                                        ▼
 ComfyUI node  ◀── Ollama local instance  ◀── ollama create ── Unified GGUF (q8_0)
```

**Consequence:** every LoRA becomes its own independent Ollama model. There is no runtime
hot-swapping — to ship a new adapter, you re-run the merge→GGUF→create pipeline and get a new tag.

---

## 2. Deployment pipeline

Fully automated by `convert_and_upload.sh` (+ `scripts/merge_and_convert_gguf.py`). Override the
target via env vars (`MODEL_BASENAME`, `BASE_MODEL_ID`, `LORA_ADAPTER_DIR`, `HF_REPO_ID`,
`OLLAMA_MODEL_NAME`, `MODELFILE`). The steps it runs:

**Step 1 — Merge in high precision.** `bfloat16`, `device_map="auto"` (GPU). The script then merges
and saves the tokenizer, and applies the `extra_special_tokens` workaround (see Limitations).
VRAM-constrained? Switch to `device_map="cpu"` to merge a 7B/8B in system RAM, then quantize.

```python
base   = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="auto")
merged = PeftModel.from_pretrained(base, lora_dir).merge_and_unload()
merged.save_pretrained(merged_dir)
AutoTokenizer.from_pretrained(base_id).save_pretrained(merged_dir)
```

**Step 2 — Convert + quantize to GGUF** (q8_0 = near-lossless for prompt generation):

```bash
python llama.cpp/convert_hf_to_gguf.py "$MERGED_MODEL_DIR" \
    --outfile "$GGUF_OUTPUT_FILE" --outtype q8_0
```

**Step 3 — Configure the Modelfile** (`Modelfile.image-v1`). The `SYSTEM` forces raw output; the
`TEMPLATE` auto-wraps the ComfyUI input in the training instruction format; `stop` tokens prevent
rambling:

```dockerfile
FROM ./outputs/qwen2-5-7b-image-v2-q8.gguf
SYSTEM """You are an expert image prompt generator for Stable Diffusion / FLUX. When given a style
name or scene description, output the image prompt followed by an optional negative prompt on a new
line prefixed with 'Negative:'. No labels, no preamble, no command-line flags."""
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
Generate a detailed image prompt in the style of '{{ .Prompt }}'.<|im_end|>
<|im_start|>assistant
"""
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.2
PARAMETER num_ctx 1024
PARAMETER num_predict 250
PARAMETER stop <|im_end|>
PARAMETER stop <|im_start|>
```

**Step 4 — Build the Ollama model:**

```bash
ollama create qwen2-5-7b-image-prompt:v2 -f Modelfile.image-v1
```

**Step 5 — Connect in ComfyUI.** After the script succeeds, wait ~60 s (Ollama list cache TTL) or
restart ComfyUI, then:
1. The tag appears in the **Limbicnation Prompt Generator** node's `model_name` dropdown.
2. Pipe `description` straight in — the Modelfile `TEMPLATE` auto-prepends the instruction, so bare
   style names (`Cinematic Noir`) match the LoRA's training distribution.
3. Route the output through the **Negative Prompt Extractor** node to split the body from the
   `Negative:` line.
4. Feed the positive/negative into the CLIP/T5 encoder of your sampler.

Smoke test before touching ComfyUI:

```bash
ollama run qwen2-5-7b-image-prompt:v2 "Cinematic Noir"
```

---

## 3. Node ↔ template routing

This is the part most guides miss. ComfyUI Ollama nodes hit one of two endpoints, and Ollama exposes
a **different set of template variables** to each:

| Node type | Ollama endpoint | Template vars populated |
|-----------|-----------------|-------------------------|
| Generate-style (repo's `ComfyUI-PromptGenerator`, stavsap `OllamaGenerateV2`) | `/api/generate` | `.System`, `.Prompt`, `.Response` — **never `.Messages`** |
| Chat-style (stavsap `OllamaChat`) | `/api/chat` | `.Messages`, `.System` |

A `TEMPLATE` written against `.Prompt` renders under **both** endpoints → portable. A template
written against `{{ range .Messages }}` renders **empty** under `/api/generate` (no message list),
so the user's text vanishes **with no error**.

| Model | Modelfile | Template style | Generate node | Chat node |
|-------|-----------|----------------|---------------|-----------|
| `image-v1`/`image-v2` | `Modelfile.image-v1` | `{{ .Prompt }}` auto-wraps | ✅ bare style/scene | ✅ |
| `dual-stream` | `Modelfile.dual-stream` | `{{ if .Prompt }}` no auto-prefix | ✅ user types dialect prefix | ✅ |
| `deforum-v7` | `Modelfile.deforum-v7` | `{{ range .Messages }}` + prefix | ❌ **prompt silently dropped** | ✅ only |

**dual-stream routing:** the dialect prefix *is* the routing signal and must vary per request — there
is no auto-prefix. Put it in the prompt text:
`Generate a FLUX (T5-XXL) image prompt for: <concept>` /
`Generate an SDXL image prompt for: <concept>` /
`Write a compact descriptive caption for: <concept>` /
`List image steering modifiers for: <concept>`.

---

## 4. Template / output rules (clean ingestion)

- **No preamble.** The LLM must not emit "Here is your prompt:". The Modelfile `SYSTEM` states this
  explicitly ("No labels, no preamble, no command-line flags").
- **Register stop tokens.** Without `<|im_end|>` / `<|im_start|>` (and the model EOS), Ollama runs to
  `num_predict` and rambles/hallucinates.
- **Auto-prefixing via `TEMPLATE`.** Input `Steampunk` → the template wraps it as
  `Generate a detailed image prompt in the style of 'Steampunk'.`, firing the right generation
  weights without manual orchestration in ComfyUI.
- **System field on the node:** paste the Modelfile `SYSTEM` (or leave it truly empty) — a non-empty
  blank in the node can overwrite the Modelfile system prompt.
- **Replicate sampling params** in the node's options — they override the Modelfile `PARAMETER`s:
  image `temperature 0.8 / top_p 0.9 / repeat_penalty 1.2 / num_ctx 1024 / num_predict 250`;
  dual-stream `num_ctx 2048`.
- **Disable context carry-over** (`keep_context = False` / fresh chat history) — stale context bleeds
  the previous concept into the next prompt, the cross-contamination class the training fought.
- **Match the dialect to the encoder downstream** (FLUX/T5-XXL vs SDXL/CLIP).

---

## 5. Limitations & failures

| Category | Technical cause | Mitigation |
|----------|-----------------|------------|
| No dynamic swapping | Ollama can't hot-swap raw `.safetensors` via API | Pre-merge each LoRA and compile its own GGUF model (this pipeline) |
| Silent prompt drop | `/api/generate` + a `{{ range .Messages }}` template → empty render, no error | Keep image Modelfiles on `.Prompt`; drive `.Messages` models with a chat node only |
| Tokenizer serialization | HF serializes `extra_special_tokens` as a list → llama.cpp conversion error | Strip it when list-typed before converting (handled in `convert_and_upload.sh`) |
| Model rambling | Missing chat template / stop params → writes until `num_predict` | Register `<|im_end|>`/`<|im_start|>` stops; keep the `TEMPLATE` |
| Options override silently | Node `options` win over Modelfile `PARAMETER`s | Replicate temp/top_p/repeat_penalty/num_ctx/num_predict in the node |
| VRAM bottleneck | Merging 7B/8B in bf16 needs lots of GPU memory | Merge on CPU (`device_map="cpu"`) in bfloat16, then quantize |
| Size mismatch | Applying a LoRA to a wrong-size base (3B adapter on 7B base) | Check `base_model_name_or_path` in the adapter's `adapter_config.json` matches the base |
| Context truncation | Long ComfyUI input + system exceeds `num_ctx` | Raise `num_ctx` in the node options |
| Missing tag | Model not created on the Ollama host ComfyUI points at | Run `convert_and_upload.sh` step 3 there; `ollama list` to confirm |

---

## 6. Quick verification

```bash
ollama list | grep image                          # confirm the tag exists
ollama show --modelfile qwen2-5-7b-image-prompt:v2 # confirm TEMPLATE uses .Prompt (not .Messages)

# Both endpoints must succeed for a .Prompt model (the compatibility claim):
curl -s localhost:11434/api/generate -d '{"model":"qwen2-5-7b-image-prompt:v2","prompt":"cyberpunk alley","stream":false}'
curl -s localhost:11434/api/chat     -d '{"model":"qwen2-5-7b-image-prompt:v2","messages":[{"role":"user","content":"cyberpunk alley"}],"stream":false}'
```

Contrast (demonstrates the trap): the same `/api/generate` call against the `.Messages`-template
`qwen3-4b-deforum-prompt:v7` returns empty/garbage, while its `/api/chat` call works.

**End-to-end:** load the **Limbicnation Prompt Generator** node, select the image tag, feed a bare
concept (or a dialect-prefixed concept for dual-stream), confirm a clean prompt with no
preamble/labels flows through the Negative Prompt Extractor into the CLIP/T5 encode node.
