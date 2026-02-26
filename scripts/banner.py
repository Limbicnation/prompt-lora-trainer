#!/usr/bin/env python3
"""
ASCII Art Banner for Prompt LoRA Trainer CLI

Displays the Option A – Clean Professional banner at startup.
Import and call print_banner() from any script.
"""


BANNER = r"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   ◆  P R O M P T   L O R A   T R A I N E R  ◆                        ║
║   ──────────────────────────────────────────────────────────────────   ║
║   QLoRA Fine-Tuning for Video Diffusion Prompts                        ║
║   Qwen3-4B/8B  ·  RTX 4090  ·  WandB  ·  v7                          ║
║                                                                        ║
║   Targets: LTX-Video · WanVideo · ComfyUI                             ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


def print_banner():
    """Print the CLI banner to stdout."""
    print(BANNER)


if __name__ == "__main__":
    print_banner()
