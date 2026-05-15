#!/bin/bash
#SBATCH --job-name=job
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-gpu=224GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
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
cd /storage/home/hcoda1/6/haltahan6/scratch/unibench/tests/haider

python eval_openapp.py --output_dir=$1 --idx=$2 --num_workers=8 --benchmark_id=$SLURM_ARRAY_TASK_ID
