#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --array=1-1
#SBATCH --output=/home/druhe/rail1/src/cnf/slurm-%j.out
cd /home/druhe/rail1/src/cnf/
git pull
git checkout 8d8kpmvb
source env/activate.sh
WANDB_ENABLED=TRUE wandb agent druhe/cnf/8d8kpmvb