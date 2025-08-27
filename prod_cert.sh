#!/bin/bash
#SBATCH -A IscrC_ADGA
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --time 12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=cert_vae
#SBATCH -o run_cert_vae1.out
#SBATCH -e run_cert_vae1.err

source ~/.bashrc
conda activate cert

export LD_LIBRARY_PATH=/leonardo/home/userexternal/fgiacoma/micromamba/envs/cert/lib:$LD_LIBRARY_PATH

srun python Cert_algs/certify_vae.py