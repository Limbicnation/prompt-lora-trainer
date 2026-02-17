• ✅ Verification Complete - Ready for Full Run

  Current State

   Component    Status            Details
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Script       ✅ Ready          scripts/build_dataset_v3.py committed with fixes
   Dry-run      ✅ Passed         Sarah 0.7%, chiaroscuro 9.1%, 4,326 rows
   Git branch   ✅ Clean          feat/dataset-v3-pipeline with 2 commits
   Ollama       ⚠️ Action needed   qwen3:4b not pulled yet

  Pre-Run Checklist

  Before running the full pipeline, ensure:

# 1. Pull the required model (2.5 GB)

  ollama pull qwen3:4b

# 2. Verify HF_TOKEN is set

  echo $HF_TOKEN  # Should show your token

# 3. Run in persistent session (tmux/screen recommended)

  tmux new -s dataset-v3-build

  Full Pipeline Command

  cd /home/gero/GitHub/DeepLearning_Lab/prompt-lora-trainer

  conda run -n prompt-lora-trainer uv run scripts/build_dataset_v3.py \
    --target Limbicnation/deforum-prompt-lora-dataset-v3 \
    --seed 42 \
    --synthesize-with-ollama \
    --ollama-model qwen3:4b \
    --batch-size 10

  Expected Timeline

   Phase                   Duration     Output
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Stage 1 (v2 clean)      ~5 sec       4,860 rows
   Stage 2 scoring         ~10 sec      1,330 selected
   Stage 2 synthesis       ~20-30 min   1,330 LLM calls
   Stage 3 chunking        ~2 sec       700 chunks
   Stage 3 synthesis       ~10-15 min   700 LLM calls
   Stage 4 dedupe/upload   ~10 sec      ~5,500 final rows
   Total                   ~30-45 min   Dataset on Hub

  Post-Build Steps

  After successful build:

# 1. Update config

# Edit configs/sft_qwen3_4b_deforum_v3.yaml

# dataset_id: "Limbicnation/deforum-prompt-lora-dataset-v3"

# 2. Dry-run training

  conda run -n prompt-lora-trainer python scripts/train_sft.py \
    --config configs/sft_qwen3_4b_deforum_v3.yaml --dry-run

# 3. Full training (1-3 hours)

  conda run -n prompt-lora-trainer python scripts/train_sft.py \
    --config configs/sft_qwen3_4b_deforum_v3.yaml

  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  🚀 Ready to proceed. Just pull qwen3:4b first, then run the full pipeline in a tmux session.
