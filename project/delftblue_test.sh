#!/bin/sh

#SBATCH --job-name="train_rl_models"
#SBATCH --partition=gpu-a100
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --time=20:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=7GB

# Alternatively, for partition we can use 'gpu', less powerful (4x with 32 GB RAM instead of 4x with 80 GB RAM)

module load 2023r1
module load openmpi
module load python
module load py-matplotlib
module load py-torch
module load py-pip

pip install imitation
pip install stable-baselines3[extra]

srun ./lunar_lander.py > output.log