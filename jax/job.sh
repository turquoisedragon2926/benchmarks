#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=2 #equal to -N 1
#SBATCH --tasks-per-node=2
#SBATCH --exclusive
#SBATCH --job-name=jax-fft-test
#SBATCH --gpus=4
#SBATCH --output output/slurm-%j.out

nvidia-smi

source $DATA/venv-jax/bin/activate

cd ~/jax-testing/

#export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export XLA_PYTHON_CLIENT_ALLOCATOR=platform

srun --output "output/slurm-%2j-%2t.out" python -u main.py
