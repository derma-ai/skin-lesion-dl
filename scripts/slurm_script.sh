#!/bin/bash -l

#SBATCH --job-name=weighted_focal
#SBATCH --output=res.txt
#SBATCH --error=res.err

# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=8
# #SBATCH --gpus=2


source activate idp_env
srun ./train.sh 0
