#!/bin/bash -l
#SBATCH -p a100 
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --time=2-00:00:00
#SBATCH --mem=30GB
#SBATCH --job-name="video_background"
#SBATCH --gres=gpu:1,tmpfs:50G # generic resource required (here requires 1 GPU)
module load Anaconda3/2023.03
conda activate trading

python run.py