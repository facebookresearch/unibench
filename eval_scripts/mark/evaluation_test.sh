#!/bin/bash
#SBATCH --job-name=unibench_mark_test
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=./scripts_log/%j.out
#SBATCH --account=gts-rmurty7-paid

# No GPU needed — all models call external APIs.

. "/storage/home/hcoda1/6/haltahan6/r-rmurty7-0/haider/miniconda/etc/profile.d/conda.sh"
conda activate flash
which python
export PYTHONPATH="${PYTHONPATH}:/storage/home/hcoda1/6/haltahan6/scratch/unibench"
export HF_HOME=/storage/home/hcoda1/6/haltahan6/scratch/.cache/hf_home
export HUGGINGFACE_HUB_CACHE=/storage/home/hcoda1/6/haltahan6/scratch/.cache/hf
export UNIBENCH_HUB=/storage/home/hcoda1/6/haltahan6/scratch/.cache/unibench
export TORCH_HOME=/storage/home/hcoda1/6/haltahan6/scratch/.cache/torch

# API keys are forwarded by run.sh / test_run.sh via --export; nothing to set here.

# Arguments:
#   $1  output_dir
#   $2  model name (string) or model index (integer)
#   $3  mode  (relation | classification | vqa | all)  [default: vqa]
#   $4  benchmark_id  (only used for classification/all)

cd /storage/home/hcoda1/6/haltahan6/scratch/unibench/eval_scripts/mark

OUTPUT_DIR=$1
MODEL=$2
MODE=${3:-vqa}
BENCHMARK_ID=${4:-2}

# Detect whether $MODEL is a number (index) or a name
if [[ "$MODEL" =~ ^[0-9]+$ ]]; then
    MODEL_ARG="--idx=${MODEL}"
else
    MODEL_ARG="--model_name=${MODEL}"
fi

if [[ "$MODE" == "classification" || "$MODE" == "all" ]]; then
    python eval.py \
        --output_dir="${OUTPUT_DIR}" \
        ${MODEL_ARG} \
        --num_workers=4 \
        --mode="${MODE}" \
        --benchmark_id="${BENCHMARK_ID}"
else
    python eval.py \
        --output_dir="${OUTPUT_DIR}" \
        ${MODEL_ARG} \
        --num_workers=4 \
        --mode="${MODE}"
fi
