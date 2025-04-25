#!/bin/bash
#SBATCH --job-name=job
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-gpu=64GB
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=./scripts_log/%A_%a.out
#SBATCH --partition=devlab

source /private/home/marksibrahim/Projects/Unibench/unibench/unibench2/bin/activate
which python
cd /private/home/marksibrahim/Projects/Unibench/unibench/test

python run_siglip2_evaluation.py --task_name=$1 --idx=$SLURM_ARRAY_TASK_ID
