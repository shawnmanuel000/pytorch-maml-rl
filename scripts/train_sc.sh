#!/bin/bash

# Slurm sbatch options
#SBATCH -o mpi/mpi_%j.log
# SBATCH -n 2
#SBATCH -N 1
#SBATCH -p napoli-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
# SBATCH --mem=32GB

# Initialize the module command first
source ~/.bashrc
conda activate py377

# export DISPLAY=':99.0'
# Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &

echo "python -B train.py --env_name $env_name"
python -B train.py --env_name $env_name
