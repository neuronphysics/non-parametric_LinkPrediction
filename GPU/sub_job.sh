#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00
#SBATCH --account=def-jhoey

./glfm_gpu_excutable_2000_fix_sRho > log_2000_fix_sRho
