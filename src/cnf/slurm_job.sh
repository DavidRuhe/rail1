#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=120:00:00
#SBATCH --gpus-per-node=1
#SBATCH --array=1-1
#SBATCH --output=/home/druhe/rail1/src/cnf/slurm-%j.out
cd /home/druhe/rail1/src/cnf/
git checkout tr0lz3ky
source ./activate.sh
WANDB_ENABLED=TRUE wandb agent druhe/cnf/tr0lz3ky