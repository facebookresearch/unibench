#!/bin/bash
# test_run.sh – submit a single quick test job for gpt_5_4_genai_responses.
# ============================================================
# EDIT THE SECTION BELOW BEFORE RUNNING
# ============================================================

# --- API keys ---
OPENAI_API_KEY=""                   # required for gpt_5_4_genai_responses
ANTHROPIC_VERTEX_PROJECT_ID=""      # only needed for claude_4_6_opus_genai_vertex
GOOGLE_API_KEY=""                   # only needed for gemini_3_1_pro_preview_fair

# --- What to run ---
MODEL_NAME="gpt_5_4_genai_responses"
MODE="vqa"   # vqa | relation | classification | all

# --- Where to write results ---
OUT_DIR="$(realpath ./script_outputs/)"

# ============================================================
# Nothing below this line should need to change
# ============================================================

RUN_DATE=$(date +%Y%m%d_%H%M%S)
LOG_DIR=./scripts_log/${RUN_DATE}
LOG_FILE=${LOG_DIR}/slurm_%j.out

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

echo "Logging to:       ${LOG_FILE}"
echo "Output directory: ${OUT_DIR}"
echo "Model:            ${MODEL_NAME}"
echo "Mode:             ${MODE}"

cd /storage/home/hcoda1/6/haltahan6/scratch/unibench/eval_scripts/mark

sbatch --output="${LOG_FILE}" \
    --export=ALL,OPENAI_API_KEY="${OPENAI_API_KEY}",ANTHROPIC_VERTEX_PROJECT_ID="${ANTHROPIC_VERTEX_PROJECT_ID}",GOOGLE_API_KEY="${GOOGLE_API_KEY}" \
    evaluation_test.sh "${OUT_DIR}" "${MODEL_NAME}" "${MODE}"

echo "Done submitting."
