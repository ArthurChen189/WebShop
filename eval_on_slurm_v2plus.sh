#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=eval_webshop_v1
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=12:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --partition=rtx6000
#SBATCH --output=webshop_logs/slurm-%j-v2plus.out
#SBATCH --error=webshop_logs/slurm-%j-v2plus.err

N_GPU=1

source /fs01/home/arthur/.zshrc
eval "$(micromamba shell hook --shell zsh)"
micromamba activate webshop-eval
cd /fs01/home/arthur/Workspace/WebShop

bash scratch/run_eval_v2plus.sh $N_GPU