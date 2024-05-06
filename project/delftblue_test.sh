#!/bin/sh

#SBATCH --job-name="delftblue_test"
#SBATCH --partition=compute
#SBATCH --account=research-eemcs-diam
#SBATCH --time=01:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB

module load 2023r1
module load openmpi
module load python
module load py-matplotlib
module load py-torch
module load py-pip

pip install imitation
pip install stable-baselines3[extra]

srun ./compare.py > output.log