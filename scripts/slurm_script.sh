#!/bin/bash

#SBATCH --job-name=processing-experiment
#SBATCH --output=res.txt
#SBATCH --error=res.err

# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=8
# #SBATCH --gpus=2

CUDA_VISIBLE_DEVICES=0

source /u/home/podszun/.bashrc
source activate idp_env
srun which python
srun train.sh
