#!/bin/bash
#SBATCH -A IscrC_ADGA
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --time 12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=vae_mnist
#SBATCH -o run_vae64.out
#SBATCH -e run_vae64.err

source ~/.bashrc
conda activate cert

export LD_LIBRARY_PATH=/leonardo/home/userexternal/fgiacoma/micromamba/envs/cert/lib:$LD_LIBRARY_PATH

srun python Models/score_based/exec_vae.py