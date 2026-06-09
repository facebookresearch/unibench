#!/bin/bash
# run.sh – submit ALL evaluation jobs (classification + relation) for all models.
#
# Classification: one SLURM array job per model; array tasks 0-44 = benchmark IDs.
# Relation:       one regular job per model covering all 8 relation benchmarks.
#
# Edit NUM_MODELS if the model list in eval.py changes (currently 35 models, idx 0-34).

NUM_MODELS=34   # last index (0-based)

# === Get current date ===
RUN_DATE=$(date +%Y%m%d_%H%M%S)

# === Define output/log dirs ===
LOG_DIR=./scripts_log/${RUN_DATE}
OUT_DIR=$(realpath ./script_outputs/)

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

LOG_FILE=${LOG_DIR}/slurm_%A_%a.out

echo "Logging to:        ${LOG_FILE}"
echo "Output directory:  ${OUT_DIR}"

cd /storage/home/hcoda1/6/haltahan6/scratch/unibench/eval_scripts/haider

echo "Submitting classification jobs for model indices 0-${NUM_MODELS} ..."
for model_idx in $(seq 0 ${NUM_MODELS}); do
    # Array tasks 0-44 map directly to CLASSIFICATION_BENCHMARKS indices
    sbatch --array=0-44 --output="${LOG_FILE}" \
        evaluation.sh "${OUT_DIR}" "${model_idx}" "classification"
done

echo "Submitting relation jobs for model indices 0-${NUM_MODELS} ..."
for model_idx in $(seq 0 ${NUM_MODELS}); do
    sbatch --output="${LOG_FILE}" \
        evaluation.sh "${OUT_DIR}" "${model_idx}" "relation"
done

echo "Submitting VQA jobs (openapps, mmmu_pro) for model indices 0-${NUM_MODELS} ..."
for model_idx in $(seq 0 ${NUM_MODELS}); do
    sbatch --output="${LOG_FILE}" \
        evaluation.sh "${OUT_DIR}" "${model_idx}" "vqa"
done

echo "Done submitting."
