#!/bin/bash

# === Get current date ===
RUN_DATE=$(date +%Y%m%d_%H%M%S)

# === Define output dirs based on date ===
LOG_DIR=./scripts_log/${RUN_DATE}
OUT_DIR=./script_outputs/

# === Create the directories ===
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

# === Define output file for SLURM ===
LOG_FILE=${LOG_DIR}/slurm_%A_%a.out  # %A: job ID, %a: array index

source /fsx-robust/marksibrahim/tmp/UniBench/unibench/.venv/bin/activate
uv pip install -U /fsx-robust/marksibrahim/tmp/UniBench/unibench[all]
cd /fsx-robust/marksibrahim/tmp/UniBench/unibench/tests/mark

unibench version

# === Submit the job ===
sbatch --output="${LOG_FILE}" --array=0-32 evaluation.sh "${OUT_DIR}"
