#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p napoli-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:00:40
#SBATCH --mem=1GB

sleep 60
