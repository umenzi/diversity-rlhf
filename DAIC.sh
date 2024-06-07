#!/bin/sh
#SBATCH --partition=general   # Request partition. Default is 'general'
#SBATCH --qos=medium          # Request Quality of Service. 'medium' means max runtime is 1.5 days
#SBATCH --time=36:00:00       # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1            # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=8     # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=16GB            # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --mail-type=END       # Set mail type to 'END' to receive a mail when the job finishes.
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --output=slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm_%j.err  # Set name of error log. %j is the Slurm jobId

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

/usr/bin/nvidia-smi # Check sbatch settings are working (it should show the GPU that you requested)

# Remaining job commands go below here. For example, to run python code that makes use of GPU resources:
module use /opt/insy/modulefiles          # Use DAIC INSY software collection
module load cuda/12.4 cudnn/12-8.9.1.23 # Load certain versions of cuda and cudnn

# Some debugging logs
which python3 1>&2  # Write path to Python binary to standard error
python3 --version   # Write Python version to standard error

pip install -r requirements.txt

# Run your script with the `srun` command:
srun python3 src/train.py
