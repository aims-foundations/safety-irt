#!/bin/bash
# Usage: ./collect_data.sh
# Reproduces the full data collection pipeline from the paper:
#   1. Download & reshape MultiJail dataset
#   2. Collect test-taker responses (all model families)
#   3. Merge results and run batch grading
#
# Prerequisites:
#   - Python virtual environment with dependencies installed (see reproduce.sh)
#   - API keys exported as environment variables (see below)
#
# Required API keys:
#   export OPENAI_API_KEY="..."      # GPT models + batch grading judge
#   export ANTHROPIC_API_KEY="..."   # Claude models
#   export GOOGLE_API_KEY="..."      # Gemini models
#   export XAI_API_KEY="..."         # Grok models
#   export DEEPSEEK_API_KEY="..."    # DeepSeek models
#
# Optional (local Qwen via vLLM):
#   export VLLM_BASE_URL="http://localhost:8234/v1"

set -e
cd "$(dirname "$0")"

RUNNER="python -m data_curation.test_takers"

# ─── Preflight checks ─────────────────────────────────────────────────────────

echo "=== Preflight checks ==="

missing_keys=()
[ -z "$OPENAI_API_KEY" ]    && missing_keys+=("OPENAI_API_KEY")
[ -z "$ANTHROPIC_API_KEY" ] && missing_keys+=("ANTHROPIC_API_KEY")
[ -z "$GOOGLE_API_KEY" ]    && missing_keys+=("GOOGLE_API_KEY")
[ -z "$XAI_API_KEY" ]       && missing_keys+=("XAI_API_KEY")
[ -z "$DEEPSEEK_API_KEY" ]  && missing_keys+=("DEEPSEEK_API_KEY")

if [ ${#missing_keys[@]} -gt 0 ]; then
    echo "WARNING: Missing API keys: ${missing_keys[*]}"
    echo "Configs requiring those keys will be skipped."
fi

# ─── Step 1: Download MultiJail dataset ────────────────────────────────────────

echo ""
echo "=== Step 1: Downloading MultiJail dataset ==="
python -m data_curation.shared.multijail
# Output: multijail.csv (315 prompts x 10 languages = 3,150 rows)

# ─── Step 2: Collect test-taker responses ──────────────────────────────────────

echo ""
echo "=== Step 2: Collecting test-taker responses ==="

# Cloud API configs and their required env vars
declare -A CONFIG_KEYS
CONFIG_KEYS[gpt]="OPENAI_API_KEY"
CONFIG_KEYS[grok]="XAI_API_KEY"
CONFIG_KEYS[deepseek]="DEEPSEEK_API_KEY"
CONFIG_KEYS[claude_3]="ANTHROPIC_API_KEY"
CONFIG_KEYS[claude_4_5_low_creativity]="ANTHROPIC_API_KEY"
CONFIG_KEYS[claude_4_5_high_risk]="ANTHROPIC_API_KEY"
CONFIG_KEYS[gemini]="GOOGLE_API_KEY"

# Run order (longest first to parallelize wall-clock time if desired)
CONFIGS=(gpt gemini grok claude_3 claude_4_5_low_creativity claude_4_5_high_risk deepseek)

for config in "${CONFIGS[@]}"; do
    required_key="${CONFIG_KEYS[$config]}"
    if [ -z "${!required_key}" ]; then
        echo "--- Skipping $config (missing $required_key) ---"
        continue
    fi
    echo ""
    echo "--- Running $config ---"
    $RUNNER --config "$config" --input multijail.csv
done

# Optional: local Qwen via vLLM (requires vLLM server running)
if [ -n "$VLLM_BASE_URL" ]; then
    echo ""
    echo "--- Running qwen (local vLLM) ---"
    $RUNNER --config qwen --input multijail.csv
else
    echo ""
    echo "--- Skipping qwen (VLLM_BASE_URL not set) ---"
fi

# ─── Step 3: Merge all results ────────────────────────────────────────────────

echo ""
echo "=== Step 3: Merging results ==="

RESULT_FILES=()
for f in *_results.csv; do
    [ -f "$f" ] && RESULT_FILES+=("$f")
done

if [ ${#RESULT_FILES[@]} -eq 0 ]; then
    echo "ERROR: No result files found."
    exit 1
fi

echo "Found result files: ${RESULT_FILES[*]}"
python -m data_curation.shared.postprocessing merge \
    --files "${RESULT_FILES[@]}" \
    --output all_results_merged.csv

# ─── Step 4: Batch grading with GPT-5.2 ───────────────────────────────────────

echo ""
echo "=== Step 4: Batch grading setup ==="

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY required for batch grading. Stopping here."
    echo "Merged results saved to all_results_merged.csv"
    exit 1
fi

# Add original prompts back for grading context
python -m data_curation.batch_grading add-prompts \
    --prompts multijail.csv \
    --results all_results_merged.csv \
    --output all_results_with_prompts.csv

# Create batch JSONL
python -m data_curation.batch_grading create-jsonl \
    --input all_results_with_prompts.csv \
    --output grading_batch.jsonl

# Estimate cost
python -m data_curation.batch_grading estimate-cost \
    --input all_results_with_prompts.csv

echo ""
echo "=== Next steps (manual) ==="
echo "1. Upload and submit the batch job:"
echo "   python -m data_curation.batch_grading upload --file grading_batch.jsonl"
echo "   python -m data_curation.batch_grading submit --file-id <FILE_ID>"
echo ""
echo "2. Check status and retrieve results:"
echo "   python -m data_curation.batch_grading check --batch-id <BATCH_ID>"
echo "   python -m data_curation.batch_grading retrieve --batch-id <BATCH_ID> --output grading_results.jsonl"
echo ""
echo "3. Merge scores and analyze:"
echo "   python -m data_curation.batch_grading merge-results --original all_results_with_prompts.csv --results grading_results.jsonl --output graded.csv"
echo "   python -m data_curation.batch_grading jsr --input graded.csv"
