#!/bin/bash
# Convert trained LoRA to Ollama format and upload to Hugging Face

set -e
cd /home/gero/GitHub/DeepLearning_Lab/prompt-lora-trainer
source /home/gero/anaconda3/bin/activate prompt-lora-trainer

echo "🦙 Converting and Uploading Model"
echo "=================================="

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Please set HF_TOKEN environment variable"
    echo "   export HF_TOKEN=your_token_here"
    exit 1
fi

# Step 1: Merge LoRA
echo ""
echo "Step 1: Merging LoRA adapter..."
python << 'PYTHON'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base, "./outputs/qwen3-4b-deforum-prompt-lora")
merged = model.merge_and_unload()
merged.save_pretrained("./outputs/qwen3-4b-deforum-merged")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", trust_remote_code=True)
tokenizer.save_pretrained("./outputs/qwen3-4b-deforum-merged")

# Fix extra_special_tokens serialization bug (transformers saves as list, expects dict on reload)
# These tokens are already in tokenizer.json added_tokens, so removing the field is safe.
import json
config_path = "./outputs/qwen3-4b-deforum-merged/tokenizer_config.json"
with open(config_path, "r") as f:
    config = json.load(f)
if "extra_special_tokens" in config and isinstance(config["extra_special_tokens"], list):
    del config["extra_special_tokens"]
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print("  Fixed: removed list-type extra_special_tokens from tokenizer_config.json")

print("✅ Merged!")
PYTHON

# Step 2: Convert to GGUF
echo ""
echo "Step 2: Converting to GGUF..."
if [ ! -d "./llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
fi
pip install -q -r llama.cpp/requirements.txt 2>/dev/null || true

python llama.cpp/convert_hf_to_gguf.py \
    ./outputs/qwen3-4b-deforum-merged \
    --outfile ./outputs/qwen3-4b-deforum-q8.gguf \
    --outtype q8_0

echo "✅ GGUF created: $(du -h ./outputs/qwen3-4b-deforum-q8.gguf | cut -f1)"

# Step 3: Create Ollama model
echo ""
echo "Step 3: Creating Ollama model..."
cat > ./outputs/Modelfile << 'EOF'
FROM ./qwen3-4b-deforum-q8.gguf

SYSTEM """You are an expert AI Video Prompt Engineer. Generate cinematic video prompts optimized for ComfyUI, LTX-Video, and WanVideo diffusion models."""

PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER num_ctx 2048
PARAMETER stop <|im_end|>
EOF

cd ./outputs
ollama create qwen3-4b-deforum-prompt -f Modelfile 2>/dev/null || echo "⚠️  Ollama not available, skipping"
cd ..

# Step 4: Upload to HF
echo ""
echo "Step 4: Uploading to Hugging Face..."
python << 'PYTHON'
import os
from huggingface_hub import HfApi, create_repo

api = HfApi()
token = os.environ["HF_TOKEN"]
repo_id = "Limbicnation/qwen3-4b-deforum-prompt-lora"

# Create repo
create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

# Upload LoRA adapter
api.upload_folder(
    folder_path="./outputs/qwen3-4b-deforum-prompt-lora",
    repo_id=repo_id,
    token=token
)
print("✅ LoRA adapter uploaded")

# Upload GGUF
api.upload_file(
    path_or_fileobj="./outputs/qwen3-4b-deforum-q8.gguf",
    path_in_repo="qwen3-4b-deforum-q8.gguf",
    repo_id=repo_id,
    token=token
)
print("✅ GGUF uploaded")

print(f"\n🎉 Model available at: https://huggingface.co/{repo_id}")
PYTHON

echo ""
echo "=================================="
echo "Done! Use: ollama run qwen3-4b-deforum-prompt"
