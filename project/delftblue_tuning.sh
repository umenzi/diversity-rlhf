#!/bin/sh

#SBATCH --job-name="train_rl_models"
#SBATCH --partition=gpu
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --time=20:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=7GB


module load 2023r1
module load openmpi
module load python
module load py-matplotlib
module load py-torch
module load py-pip

pip install imitation
pip install optuna
pip install stable-baselines3[extra]

srun ./tuning/lunar/tuning_ppo.py > output.log