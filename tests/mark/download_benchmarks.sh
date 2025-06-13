#!/bin/bash
#SBATCH --job-name=job
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-gpu=64GB
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --output=./scripts_log/%A_%a.out
#SBATCH --partition=devlab

source /private/home/marksibrahim/Projects/Unibench/unibench/unibench2/bin/activate
which python
cd /private/home/marksibrahim/Projects/Unibench/unibench/tests/mark

# export HF_HOME=/storage/home/hcoda1/6/haltahan6/p-rmurty7-0/haider/.cache/hf_home
# export HUGGINGFACE_HUB_CACHE=/storage/home/hcoda1/6/haltahan6/p-rmurty7-0/haider/.cache/hf
# export UNIBENCH_HUB=/storage/home/hcoda1/6/haltahan6/p-rmurty7-0/haider/.cache/unibench
# export TORCH_HOME=/storage/home/hcoda1/6/haltahan6/p-rmurty7-0/haider/.cache/torch

unibench download_benchmarks