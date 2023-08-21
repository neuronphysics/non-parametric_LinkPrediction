#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00
#SBATCH --account=def-jhoey

./glfm_gpu_excutable 0.005 0.2 0.1 10 20 3 0.4 0.0025 test_1 > log_test_1
