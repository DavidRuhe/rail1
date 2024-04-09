#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --array=1-1
#SBATCH --output=/home/druhe58/rail1/src/cnf/slurm-%j.out
cd /home/druhe58/rail1/src/cnf/
git pull
git checkout 7o5t184n
source env/activate.sh
WANDB_ENABLED=TRUE wandb agent druhe/cnf/7o5t184n