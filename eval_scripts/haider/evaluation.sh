#!/bin/bash
#SBATCH --job-name=unibench
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-gpu=224GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --time=24:00:00
#SBATCH --output=./scripts_log/%A_%a.out
#SBATCH --account=gts-rmurty7-paid

. "/storage/home/hcoda1/6/haltahan6/r-rmurty7-0/haider/miniconda/etc/profile.d/conda.sh"
conda activate flash
which python
export PYTHONPATH="${PYTHONPATH}:/storage/home/hcoda1/6/haltahan6/scratch/unibench"
export HF_HOME=/storage/home/hcoda1/6/haltahan6/scratch/.cache/hf_home
export HUGGINGFACE_HUB_CACHE=/storage/home/hcoda1/6/haltahan6/scratch/.cache/hf
export UNIBENCH_HUB=/storage/home/hcoda1/6/haltahan6/scratch/.cache/unibench
export TORCH_HOME=/storage/home/hcoda1/6/haltahan6/scratch/.cache/torch

# Arguments:
#   $1  output_dir
#   $2  model idx
#   $3  mode  (relation | classification | all)
#
# When mode includes 'classification', SLURM_ARRAY_TASK_ID is used as benchmark_id.

cd /storage/home/hcoda1/6/haltahan6/scratch/unibench/eval_scripts/haider

OUTPUT_DIR=$1
MODEL=$2
MODE=${3:-relation}

# Detect whether $MODEL is a number (index) or a name
if [[ "$MODEL" =~ ^[0-9]+$ ]]; then
    MODEL_ARG="--idx=${MODEL}"
else
    MODEL_ARG="--model_name=${MODEL}"
fi

if [[ "$MODE" == "classification" || "$MODE" == "all" ]]; then
    python eval.py \
        --output_dir="$OUTPUT_DIR" \
        ${MODEL_ARG} \
        --num_workers=8 \
        --mode="$MODE" \
        --benchmark_id="$SLURM_ARRAY_TASK_ID"
else
    python eval.py \
        --output_dir="$OUTPUT_DIR" \
        ${MODEL_ARG} \
        --num_workers=8 \
        --mode="$MODE"
fi
