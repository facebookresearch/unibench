#!/bin/bash
# test_run.sh – submit a single SLURM job for one model and one benchmark.
#
# Edit the variables below to change what is tested:
#   MODEL_NAME    – explicit model name (overrides MODEL_IDX when set)
#   MODEL_IDX     – index into MODELS list in eval.py  (used when MODEL_NAME is empty)
#   MODE          – relation | classification | vqa | all
#   BENCHMARK_ID  – index into CLASSIFICATION_BENCHMARKS (only used for classification/all)

MODEL_NAME="clip_vitL14"   # set to "" to use MODEL_IDX instead
MODEL_IDX=0
MODE="vqa"
BENCHMARK_ID=2             # 2 = cifar10 (unused for vqa mode)

# VQA test: also submit a vllm model job alongside the CLIP job
VLLM_MODEL="qwen_2_5_3b"

# === Get current date ===
RUN_DATE=$(date +%Y%m%d_%H%M%S)

# === Define output/log dirs ===
LOG_DIR=./scripts_log/${RUN_DATE}
OUT_DIR=$(realpath ./script_outputs/)

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

LOG_FILE=${LOG_DIR}/slurm_%j.out

echo "Logging to:       ${LOG_FILE}"
echo "Output directory: ${OUT_DIR}"
echo "Mode:             ${MODE}"

cd /storage/home/hcoda1/6/haltahan6/scratch/unibench/eval_scripts

# ── Job 1: CLIP model ─────────────────────────────────────────────────────────
echo "Submitting CLIP job: ${MODEL_NAME:-idx=${MODEL_IDX}}"
if [[ -n "$MODEL_NAME" ]]; then
    sbatch --output="${LOG_FILE}" \
        evaluation_test.sh "${OUT_DIR}" "${MODEL_NAME}" "${MODE}" "${BENCHMARK_ID}"
else
    sbatch --output="${LOG_FILE}" \
        evaluation_test.sh "${OUT_DIR}" "${MODEL_IDX}" "${MODE}" "${BENCHMARK_ID}"
fi

# ── Job 2: vllm model ─────────────────────────────────────────────────────────
echo "Submitting vllm job: ${VLLM_MODEL}"
sbatch --output="${LOG_FILE}" \
    evaluation_test.sh "${OUT_DIR}" "${VLLM_MODEL}" "${MODE}" "${BENCHMARK_ID}"

echo "Jobs submitted."
