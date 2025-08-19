#!/bin/bash

#SBATCH -p lovelace
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --job-name=VAE1

#SBATCH -o run_vae.out
#SBATCH -e run_vae.err

module load miniconda  # or module load miniconda


# Activate the conda environment
conda activate cert  # or conda activate scoreenv if your cluster supports it directly

# Navigate to the directory with your script
cd /u/f_giacomarra/repos/Latent_Distilled_Diffusion_4_Cert



python Models/score_based/exec_vae.py 