#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=120:00:00
#SBATCH --gpus-per-node=1
#SBATCH --array=1-1
#SBATCH --output=$HOME/rail1/src/cnf/slurm-%j.out
cd $HOME/rail1/src/cnf/
git checkout 750cvj28
source ./activate.sh
WANDB_ENABLED=TRUE wandb agent druhe/cnf/750cvj28