#!/bin/bash
# Environment setup for prompt-lora-trainer
# Run this script to create and configure the Conda environment

set -e

ENV_NAME="prompt-lora-trainer"
PYTHON_VERSION="3.10"

echo "🚀 Setting up $ENV_NAME environment..."

# Create conda environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "📦 Installing PyTorch with CUDA 12.4..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "📦 Installing training dependencies..."
pip install transformers>=4.40.0 accelerate>=0.28.0 datasets>=2.18.0
pip install peft>=0.10.0 bitsandbytes>=0.43.0
pip install trl>=0.8.0
pip install huggingface-hub>=0.22.0
pip install wandb trackio
pip install python-dotenv pyyaml tqdm

echo "📦 Installing development tools..."
pip install pytest ruff ipykernel

echo ""
echo "✅ Environment '$ENV_NAME' created successfully!"
echo ""
echo "To activate:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify installation:"
echo "  python -c \"import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')\""
echo ""
echo "To start training:"
echo "  python scripts/train_sft.py --config configs/sft_qwen3_4b.yaml --dry-run"
