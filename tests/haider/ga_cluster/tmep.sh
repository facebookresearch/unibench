#!/bin/bash
#SBATCH --job-name=job
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-gpu=95GB
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --time=03:00:00
#SBATCH --output=./scripts_log/%A_%a.out
#SBATCH --account=gts-rmurty7-paid

#!/bin/bash
#SBATCH --job-name=job
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-gpu=224GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --time=24:00:00
#SBATCH --output=./scripts_log/%A_%a.out
#SBATCH --account=gts-rmurty7-paid

#!/bin/bash
#SBATCH --job-name=job
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-gpu=224GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --time=24:00:00
#SBATCH --output=./scripts_log/%A_%a.out
#SBATCH --account=gts-rmurty7-paid