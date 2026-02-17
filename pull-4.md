Summary of Changes
Hello @Limbicnation, I'm Gemini Code Assist1! I'm currently reviewing this pull request and will post my feedback shortly. In the meantime, here's a summary to help you and other reviewers quickly get up to speed!

This pull request introduces the third iteration of the Deforum prompt generation pipeline, significantly enhancing the quality and diversity of the training data and refining the model's inference capabilities. It establishes a robust data processing workflow, configures a specialized Qwen3-4B model for cinematic video prompt generation, and updates the training setup for improved monitoring and evaluation. The overall goal is to produce more sophisticated and varied outputs for the 'De Forum Art Film' aesthetic.

Highlights
New Dataset Pipeline (v3): Introduced a comprehensive Python script (scripts/build_dataset_v3.py) to construct the Deforum Prompt Dataset v3. This pipeline cleans the existing v2 dataset, enriches it with visually dense passages from Creative Writing ShareGPT and Gutenberg Sci-Fi sources, diversifies common terms like 'Sarah' and 'chiaroscuro', and performs fuzzy deduplication before pushing the final dataset to HuggingFace Hub.
Qwen3-4B Model Configuration: Added a new Ollama Modelfile (Modelfile.deforum-v3) for the Qwen3-4B Deforum Prompt LoRA v3. This Modelfile includes a specialized system prompt designed to generate cinematic video prompts adhering to the 'De Forum Art Film' aesthetic, complete with visual style, camera work, and technical parameters.
Supervised Fine-Tuning (SFT) Configuration: Provided a new SFT configuration file (configs/sft_qwen3_4b_deforum_v3.yaml) for training the Qwen3-4B model using the v3 dataset. This configuration specifies parameters for the model, dataset, LoRA, quantization, training, optimization, and evaluation, with monitoring configured via WandB.
Monitoring Integration: Modified the SFT training script (scripts/train_sft.py) to include optional integration for trackio monitoring, allowing for flexible reporting based on the training configuration.
Changelog
Using Gemini Code Assist
Footnotes
Review the Privacy Notices, Generative AI Prohibited Use Policy, Terms of Service, and learn how to configure Gemini Code Assist in GitHub here. Gemini can make mistakes, so double check it and use code with caution. ↩

gemini-code-assist[bot]
gemini-code-assist bot reviewed 3 hours ago
Contributor
gemini-code-assist bot
left a comment
Code Review
This pull request introduces a new v3 pipeline for building and training a Deforum prompt generation model. It adds a comprehensive dataset creation script, a new training configuration, and an Ollama Modelfile. The training script is also enhanced to support trackio for monitoring. My review identifies a critical portability issue in the Modelfile, inconsistencies in the YAML configuration, and several areas for improvement in the new dataset script, focusing on error handling, code clarity, and maintainability.

Modelfile.deforum-v3

# ollama create qwen3-deforum-v3 -f Modelfile.deforum-v3

# ollama run qwen3-deforum-v3 "Generate a cinematic video prompt for a rainy cityscape"

FROM /home/gero/GitHub/DeepLearning_Lab/prompt-lora-trainer/outputs/qwen3-4b-deforum-q8-v3.1.gguf
Contributor
@gemini-code-assist
gemini-code-assist bot
3 hours ago
critical

The FROM instruction uses a hardcoded absolute path specific to your local machine (/home/gero/...). This makes the Modelfile non-portable and will cause it to fail on any other system. It should be updated to use a relative path.

FROM ./outputs/qwen3-4b-deforum-q8-v3.1.gguf
@Limbicnation Reply...
scripts/build_dataset_v3.py
Comment on lines +343 to +346
            try:
                conversations = json.loads(conversations)
            except:
                continue
Contributor
@gemini-code-assist
gemini-code-assist bot
3 hours ago
high

Using a bare except: clause is dangerous as it catches all exceptions, including system-exiting ones like SystemExit or KeyboardInterrupt. This can make the program difficult to debug and interrupt. You should catch a more specific exception. In this case, json.loads() is likely to raise json.JSONDecodeError.

Suggested change
            try:
                conversations = json.loads(conversations)
            except:
                continue
            try:
                conversations = json.loads(conversations)
            except json.JSONDecodeError:
                continue
@Limbicnation Reply...
configs/sft_qwen3_4b_deforum_v3.yaml
Comment on lines 49 to 53
output_dir: "./outputs/qwen3-4b-deforum-prompt-lora-v3"
logging_steps: 10
save_steps: 50
push_to_hub: true
hub_model_id: "Limbicnation/qwen3-4b-deforum-prompt-lora-v3"
Contributor
@gemini-code-assist
gemini-code-assist bot
3 hours ago
medium

The versioning for output paths and model IDs is inconsistent with the dataset and run name, which can cause confusion and issues with tracking experiments.

dataset_id: ...-v3.1 (line 11)
run_name: ...-v3.1 (line 67)
output_dir: ...-v3 (line 49)
hub_model_id: ...-v3 (line 53)
It's recommended to use a consistent version number (e.g., v3.1) across all related configurations for clarity.

output_dir: "./outputs/qwen3-4b-deforum-prompt-lora-v3.1"
logging_steps: 10
save_steps: 50
push_to_hub: true
hub_model_id: "Limbicnation/qwen3-4b-deforum-prompt-lora-v3.1"
@Limbicnation Reply...
configs/sft_qwen3_4b_deforum_v3.yaml
early_stopping_patience: 3

# Monitoring - WandB (reliable cloud sync)

report_to: "wandb"
Contributor
@gemini-code-assist
gemini-code-assist bot
3 hours ago
medium

There is a contradiction in the monitoring configuration. The comment at the top of the file (line 3) mentions using 'Trackio monitoring (instead of wandb)', but report_to is set to wandb. This is confusing. Given the updates in train_sft.py to support trackio, it seems the intention was to use it.

report_to: "trackio"
@Limbicnation Reply...
scripts/build_dataset_v3.py
Comment on lines +214 to +217
    # Load via DuckDB
    conn = duckdb.connect()
    if HF_TOKEN:
        conn.execute("CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN ?);", [HF_TOKEN])
Contributor
@gemini-code-assist
gemini-code-assist bot
3 hours ago
medium

The logic for connecting to DuckDB and creating the Hugging Face token secret is duplicated in stage1_clean_v2, stage2_creative_writing (lines 316-319), and stage3_gutenberg (lines 469-472). This code duplication makes maintenance harder. Consider refactoring this into a helper function that initializes and returns a configured DuckDB connection.

@Limbicnation Reply...
scripts/build_dataset_v3.py
Comment on lines +222 to +227
    try:
        df = conn.execute("SELECT * FROM read_parquet(?)", [path]).fetchdf()
    except Exception as e:
        print(f"Error loading v2 dataset: {e}")
        conn.close()
        return pd.DataFrame()
Contributor
@gemini-code-assist
gemini-code-assist bot
3 hours ago
medium

Catching a broad Exception can hide unexpected errors and make debugging more difficult. It's a best practice to catch more specific exceptions that you expect to handle. For database operations with DuckDB, you should catch duckdb.Error.

Suggested change
    try:
        df = conn.execute("SELECT *FROM read_parquet(?)", [path]).fetchdf()
    except Exception as e:
        print(f"Error loading v2 dataset: {e}")
        conn.close()
        return pd.DataFrame()
    try:
        df = conn.execute("SELECT* FROM read_parquet(?) ", [path]).fetchdf()
    except duckdb.Error as e:
        print(f"Error loading v2 dataset: {e}")
        conn.close()
        return pd.DataFrame()
@Limbicnation Reply...
scripts/build_dataset_v3.py
            # Fallback: split by single newlines if no double newlines
            paragraphs = text.split('\n')

        # Build chunks of 200-500 words using sliding window
Contributor
@gemini-code-assist
gemini-code-assist bot
3 hours ago
medium

The comment on this line mentions a "sliding window", but the implementation that follows (lines 520-542) performs chunking without overlap. The loop advances by chunk_size (i += chunk_size), not a smaller step. This is misleading and could confuse future maintainers. Please update the comment to accurately describe the chunking logic.

Suggested change
        # Build chunks of 200-500 words using sliding window
        # Build chunks of 200-500 words by splitting the text
