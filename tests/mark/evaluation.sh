#!/bin/bash
#SBATCH --job-name=unibench
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-gpu=64GB
#SBATCH --cpus-per-gpu=12
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --account=a100-memorization

source /fsx-robust/marksibrahim/tmp/UniBench/unibench/.venv/bin/activate
which python
cd /fsx-robust/marksibrahim/tmp/UniBench/unibench/tests

export HF_HOME=/fsx-robust/marksibrahim/datasets/hf_home
export HUGGINGFACE_HUB_CACHE=/fsx-robust/marksibrahim/datasets/hf
export UNIBENCH_HUB=/fsx-robust/marksibrahim/datasets/unibench
export TORCH_HOME=/fsx-robust/marksibrahim/datasets/torch

python main.py --output_dir=$1 --idx=$SLURM_ARRAY_TASK_ID --num_workers=96
