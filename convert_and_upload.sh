#!/bin/bash
# Convert trained LoRA to Ollama format and upload to Hugging Face
#
# Default behavior (no env vars set) reproduces the v7 Qwen3 pipeline.
# Override via env vars to target a different pipeline, e.g. Gemma 3n image-v1:
#
#   MODEL_BASENAME=gemma-3n-e2b-image-v1 \
#   BASE_MODEL_ID=unsloth/gemma-3n-E2B-it \
#   HF_REPO_ID=Limbicnation/gemma-3n-e2b-image-prompt-lora-v1 \
#   OLLAMA_MODEL_NAME=gemma-3n-e2b-image-prompt:v1 \
#   TRUST_REMOTE_CODE=false \
#   MODELFILE=./Modelfile.image-v1 \
#   ./convert_and_upload.sh

set -e

OUTPUT_DIR="./outputs"

# --- Configuration (env-var overridable) ---
MODEL_BASENAME="${MODEL_BASENAME:-qwen3-4b-deforum}"
BASE_MODEL_ID="${BASE_MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
LORA_ADAPTER_DIR="${LORA_ADAPTER_DIR:-$OUTPUT_DIR/${MODEL_BASENAME}-prompt-lora}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-$OUTPUT_DIR/${MODEL_BASENAME}-merged}"
GGUF_OUTPUT_FILE="${GGUF_OUTPUT_FILE:-$OUTPUT_DIR/${MODEL_BASENAME}-q8.gguf}"
GGUF_TYPE="${GGUF_TYPE:-q8_0}"
OLLAMA_MODEL_NAME="${OLLAMA_MODEL_NAME:-${MODEL_BASENAME}-prompt}"
HF_REPO_ID="${HF_REPO_ID:-Limbicnation/${MODEL_BASENAME}-prompt-lora}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"   # default true preserves Qwen3 behavior
MODELFILE="${MODELFILE:-}"                       # absolute or repo-relative path; empty = inline default
# Export so heredocs see them
export BASE_MODEL_ID LORA_ADAPTER_DIR MERGED_MODEL_DIR GGUF_OUTPUT_FILE HF_REPO_ID TRUST_REMOTE_CODE
# --- End Configuration ---

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🦙 Converting and Uploading Model"
echo "=================================="
echo "  Base model:    $BASE_MODEL_ID"
echo "  LoRA dir:      $LORA_ADAPTER_DIR"
echo "  Merged dir:    $MERGED_MODEL_DIR"
echo "  GGUF file:     $GGUF_OUTPUT_FILE"
echo "  HF repo:       $HF_REPO_ID"
echo "  Ollama name:   $OLLAMA_MODEL_NAME"
echo "  Modelfile:     ${MODELFILE:-<inline default>}"
echo ""

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Please set HF_TOKEN environment variable"
    echo "   export HF_TOKEN=your_token_here"
    exit 1
fi

# Step 1: Merge LoRA
echo ""
echo "Step 1: Merging LoRA adapter..."
python << PYTHON
import os, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_id = os.environ["BASE_MODEL_ID"]
lora_dir = os.environ["LORA_ADAPTER_DIR"]
merged_dir = os.environ["MERGED_MODEL_DIR"]
trust = os.environ["TRUST_REMOTE_CODE"].lower() == "true"

base = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=trust
)
model = PeftModel.from_pretrained(base, lora_dir)
merged = model.merge_and_unload()
merged.save_pretrained(merged_dir)
tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=trust)
tokenizer.save_pretrained(merged_dir)

# extra_special_tokens serialization workaround (defensive — no-op when not needed)
config_path = f"{merged_dir}/tokenizer_config.json"
with open(config_path) as f:
    cfg = json.load(f)
if "extra_special_tokens" in cfg and isinstance(cfg["extra_special_tokens"], list):
    del cfg["extra_special_tokens"]
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("  Fixed: removed list-type extra_special_tokens")

print("✅ Merged!")
PYTHON

# Step 2: Convert to GGUF
echo ""
echo "Step 2: Converting to GGUF..."
if [ ! -d "./llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
fi
pip install -q -r llama.cpp/requirements.txt

python llama.cpp/convert_hf_to_gguf.py \
    "$MERGED_MODEL_DIR" \
    --outfile "$GGUF_OUTPUT_FILE" \
    --outtype "$GGUF_TYPE"

echo "✅ GGUF created: $(du -h "$GGUF_OUTPUT_FILE" | cut -f1)"

# Step 3: Create Ollama model
echo ""
echo "Step 3: Creating Ollama model..."
GGUF_BASENAME="$(basename "$GGUF_OUTPUT_FILE")"
if [ -n "$MODELFILE" ] && [ -f "$MODELFILE" ]; then
    echo "  Using custom Modelfile: $MODELFILE"
    cp "$MODELFILE" ./outputs/Modelfile
else
    echo "  Using inline default Modelfile"
    cat > ./outputs/Modelfile <<EOF
FROM ./${GGUF_BASENAME}

SYSTEM """You are an expert AI Video Prompt Engineer. Generate cinematic video prompts optimized for ComfyUI, LTX-Video, and WanVideo diffusion models."""

PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER num_ctx 2048
PARAMETER stop <|im_end|>
EOF
fi

(cd ./outputs && ollama create "$OLLAMA_MODEL_NAME" -f Modelfile) 2>/dev/null || echo "⚠️  Ollama not available, skipping"

# Step 4: Upload to HF
echo ""
echo "Step 4: Uploading to Hugging Face..."
python << PYTHON
import os
from huggingface_hub import HfApi, create_repo

api = HfApi()
token = os.environ["HF_TOKEN"]
repo_id = os.environ["HF_REPO_ID"]
lora_dir = os.environ["LORA_ADAPTER_DIR"]
gguf_file = os.environ["GGUF_OUTPUT_FILE"]

create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

api.upload_folder(
    folder_path=lora_dir,
    repo_id=repo_id,
    token=token,
    ignore_patterns=["checkpoint-*/**", "checkpoint-*"],
)
print("✅ LoRA adapter uploaded")

api.upload_file(
    path_or_fileobj=gguf_file,
    path_in_repo=os.path.basename(gguf_file),
    repo_id=repo_id,
    token=token,
)
print("✅ GGUF uploaded")

print(f"\n🎉 Model available at: https://huggingface.co/{repo_id}")
PYTHON

echo ""
echo "=================================="
echo "Done! Use: ollama run $OLLAMA_MODEL_NAME"
