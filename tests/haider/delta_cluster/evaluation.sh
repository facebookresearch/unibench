#!/bin/bash
#SBATCH --job-name=job
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-gpu=228GB
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=./scripts_log/%A_%a.out
#SBATCH --account=cqj-delta-gpu
#SBATCH --partition=gpuH200x8

. "/work/nvme/beym/haltahan/miniconda3/etc/profile.d/conda.sh"
conda activate vllm
which python
export PYTHONPATH="${PYTHONPATH}:/work/nvme/beym/haltahan/unibench/unibench"
export HF_HOME=/work/hdd/cqj/haltahan/.cache/hf_home
export HUGGINGFACE_HUB_CACHE=/work/hdd/cqj/haltahan/.cache/hf
export UNIBENCH_HUB=/work/hdd/cqj/haltahan/.cache/unibench
export TORCH_HOME=/work/hdd/cqj/haltahan/.cache/torch
cd /work/nvme/beym/haltahan/unibench/tests

python main.py --output_dir=$1 --idx=$SLURM_ARRAY_TASK_ID --num_workers=8
