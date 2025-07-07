#!/bin/bash
#SBATCH --job-name=with_discharge
#SBATCH --partition=normal               # Standard GPU
#SBATCH --nodes=1                           # 70 maximum nodes available
#SBATCH --ntasks=1               # Can use up to 96 tasks per node
#SBATCH --cpus-per-task=20                   # Thread
#SBATCH --gres=gpu:full:1                     # Request 1 GPU
#SBATCH --time=12:00:00                     # Set the walltime (max: 72:00:00)
#SBATCH --mem=200000MB                       # Maximum memory per node in the cpu queue is 380GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as2023@kit.edu
#SBATCH --output=with_discharge%j.out # Standard output file.
#SBATCH --error=with_discharge%j.err  # Standard error file.

#Activate environment
module load jupyter/ai
pip install --user numba

# Optional: Check GPU availability
nvidia-smi

python ./without_discharge.py