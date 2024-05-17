#!/bin/sh

#SBATCH --job-name="train_rl_models"
#SBATCH --partition=gpu
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --time=24:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=10GB

# Note that, in an educational account, we cannot use more than 64 CPUs, 2 GPUs,
# and 185GB of memory, and a job cannot run longer than 24 hours


module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-matplotlib
module load py-torch
module load py-pip

pip install imitation
pip install optuna
pip install stable-baselines3[extra]

srun ./tuning/lunar/tuning_rlhf.py > output.log