#!/bin/bash

# === Get current date ===
RUN_DATE=$(date +%Y%m%d_%H%M%S)

# === Define output dirs based on date ===
LOG_DIR=./scripts_log/${RUN_DATE}
OUT_DIR=$(realpath ./script_outputs/)

# === Create the directories ===
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

# === Define output file for SLURM ===
LOG_FILE=${LOG_DIR}/slurm_%A_%a.out  # %A: job ID, %a: array index

echo "Logging to: ${LOG_FILE}"
echo "Output directory: ${OUT_DIR}"

# . "/storage/home/hcoda1/6/haltahan6/p-rmurty7-0/haider/anaconda3/etc/profile.d/conda.sh"
# conda activate vllm
# pip install -U /storage/home/hcoda1/6/haltahan6/scratch/unibench[all]
cd /storage/home/hcoda1/6/haltahan6/scratch/unibench/tests/haider/ga_cluster

# unibench version

# === Submit the job ===
for num_idx in {0..9}; do
    sbatch --output="${LOG_FILE}" evaluation_main.sh "${OUT_DIR}" "${num_idx}"
done