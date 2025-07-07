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

. "/storage/home/hcoda1/6/haltahan6/p-rmurty7-0/haider/anaconda3/etc/profile.d/conda.sh"
conda activate vllm
pip install -U /storage/home/hcoda1/6/haltahan6/scratch/unibench[all]
cd /storage/home/hcoda1/6/haltahan6/scratch/unibench/tests/haider

# === Submit the job ===
sbatch --output="${LOG_FILE}" --array=0-2 evaluation.sh "${OUT_DIR}"