# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Prompt LoRA Trainer - A Python project for training LoRA (Low-Rank Adaptation) models with training loops, LoRA configurations, and evaluation utilities.

## Project Status

This is a newly initialized repository. The codebase structure is yet to be implemented.

## Intended Architecture

Based on the project description, this repository will contain:
- **Training loop**: Core training logic for LoRA fine-tuning
- **LoRA configs**: Configuration management for LoRA parameters (rank, alpha, target modules)
- **Evaluation**: Model evaluation and metrics

## Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (once requirements exist)
pip install -r requirements.txt
```

## Key Technologies (Expected)

- PyTorch for model training
- PEFT (Parameter-Efficient Fine-Tuning) or similar for LoRA implementation
- Transformers library for base models
- Accelerate for distributed training support
