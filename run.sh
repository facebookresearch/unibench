#!/bin/bash
#SBATCH --job-name=job
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=./scripts_log/%A_%a.out
#SBATCH -A robust


source ~/anaconda3/bin/activate 
conda deactivate
conda activate clip
cd /data/home/haideraltahan/unibench

unibench evaluate --models=["llava_1_5_7b"] --benchmarks_dir="/fsx-checkpoints/haideraltahan/.cache/unibench/data/" --benchmarks=['imagenetc']

