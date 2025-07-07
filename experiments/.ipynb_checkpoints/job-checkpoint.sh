#!/bin/bash
#SBATCH --job-name=with_discharge
#SBATCH --partition=gpu_a100_il               # Standard GPU
#SBATCH --nodes=1                           # 70 maximum nodes available
#SBATCH --ntasks=1               # Can use up to 96 tasks per node
#SBATCH --cpus-per-task=16                   # Thread
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --time=10:00:00                     # Set the walltime (max: 72:00:00)
#SBATCH --mem=150000MB                       # Maximum memory per node in the cpu queue is 380GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as2023@kit.edu
#SBATCH --output=with_discharge%j.out # Standard output file.
#SBATCH --error=with_discharge%j.err  # Standard error file.

#Activate environment
module load jupyter/ai/2025-02-20

# Optional: Check GPU availability
nvidia-smi

python /pfs/data6/home/ka/ka_iwu/ka_as2023/lstm_backward/experiments/with_discharge.py
