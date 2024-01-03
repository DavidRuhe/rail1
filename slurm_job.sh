#!/bin/bash
#SBATCH --array=1-5
#SBATCH --output=/tmp/tmplfxio8ip/slurm-%j.out
cd /tmp/tmplfxio8ip
git checkout 123abc
source ./activate.sh
echo 'Hello, World!'