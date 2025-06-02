#!/bin/bash
#SBATCH --job-name=uniV2
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-gpu=64GB
#SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=./scripts_log/%A_%a.out
#SBATCH --partition=scavenge

which python
cd /private/home/rbalestriero/code/unibench/tests

python main.py --num_workers=80 --idx=$SLURM_ARRAY_TASK_ID