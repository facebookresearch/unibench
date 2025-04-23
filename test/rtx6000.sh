#!/bin/bash
#SBATCH --job-name=job
#SBATCH -N1 --ntasks-per-node=4
#SBATCH --mem-per-gpu=95GB
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:RTX_6000:4
#SBATCH --time=05:00:00
#SBATCH --output=./scripts_log/%A_%a.out
#SBATCH --account=gts-rmurty7-paid

. "/storage/home/hcoda1/6/haltahan6/p-rmurty7-0/haider/anaconda3/etc/profile.d/conda.sh"
conda activate uni
which python
export PYTHONPATH="${PYTHONPATH}:/storage/home/hcoda1/6/haltahan6/scratch/unibench"
export HF_HOME=/storage/home/hcoda1/6/haltahan6/p-rmurty7-0/haider/.cache/hf_home
export HUGGINGFACE_HUB_CACHE=/storage/home/hcoda1/6/haltahan6/p-rmurty7-0/haider/.cache/hf
export UNIBENCH_HUB=/storage/home/hcoda1/6/haltahan6/p-rmurty7-0/haider/.cache/unibench
export TORCH_HOME=/storage/home/hcoda1/6/haltahan6/p-rmurty7-0/haider/.cache/torch
cd /storage/home/hcoda1/6/haltahan6/scratch/unibench/test

python run_siglip2_evaluation.py --task_name=$1 --idx=$SLURM_ARRAY_TASK_ID