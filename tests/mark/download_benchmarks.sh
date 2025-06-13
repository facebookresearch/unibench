#!/bin/bash
#SBATCH --job-name=unibench
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-gpu=64GB
#SBATCH --cpus-per-gpu=12
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=./scripts_log/%A_%a.out
#SBATCH --account=a100-memorization

source /home/marksibrahim/Projects/UniBench/unibench/.venv/bin/activate
which python
cd /home/marksibrahim/Projects/UniBench/unibench/tests/mark

export HF_HOME=/fsx-robust/marksibrahim/datasets/hf_home
export HUGGINGFACE_HUB_CACHE=/fsx-robust/marksibrahim/datasets/hf
export UNIBENCH_HUB=/fsx-robust/marksibrahim/datasets/unibench
export TORCH_HOME=/fsx-robust/marksibrahim/datasets/torch

unibench download_benchmarks