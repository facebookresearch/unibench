#!/bin/bash
# run.sh – submit ALL evaluation jobs (relation + vqa) for all API models.
# ============================================================
# EDIT THE SECTION BELOW BEFORE RUNNING
# ============================================================

# --- API keys ---
OPENAI_API_KEY=""                   # required for gpt_* models
ANTHROPIC_VERTEX_PROJECT_ID=""      # required for claude_* models
GOOGLE_API_KEY=""                   # required for gemini_* models

# --- Where to write results ---
OUT_DIR="$(realpath ./script_outputs/)"

# --- Number of models (last 0-based index in MODELS list in eval.py) ---
NUM_MODELS=5

# ============================================================
# Nothing below this line should need to change
# ============================================================

RUN_DATE=$(date +%Y%m%d_%H%M%S)
LOG_DIR=./scripts_log/${RUN_DATE}
LOG_FILE=${LOG_DIR}/slurm_%A_%a.out

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

echo "Logging to:        ${LOG_FILE}"
echo "Output directory:  ${OUT_DIR}"

cd /storage/home/hcoda1/6/haltahan6/scratch/unibench/eval_scripts/mark

EXPORT_KEYS="ALL,OPENAI_API_KEY=${OPENAI_API_KEY},ANTHROPIC_VERTEX_PROJECT_ID=${ANTHROPIC_VERTEX_PROJECT_ID},GOOGLE_API_KEY=${GOOGLE_API_KEY}"

echo "Submitting relation jobs for model indices 0-${NUM_MODELS} ..."
for model_idx in $(seq 0 ${NUM_MODELS}); do
    sbatch --output="${LOG_FILE}" --export="${EXPORT_KEYS}" \
        evaluation.sh "${OUT_DIR}" "${model_idx}" "relation"
done

echo "Submitting VQA jobs (openapps, mmmu_pro) for model indices 0-${NUM_MODELS} ..."
for model_idx in $(seq 0 ${NUM_MODELS}); do
    sbatch --output="${LOG_FILE}" --export="${EXPORT_KEYS}" \
        evaluation.sh "${OUT_DIR}" "${model_idx}" "vqa"
done

echo "Done submitting."
