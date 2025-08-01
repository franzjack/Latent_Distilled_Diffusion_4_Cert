#!/bin/bash

#SBATCH -p lovelace
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --job-name=DDIM_x

#SBATCH -o run_csdi_x_test.out
#SBATCH -e run_csdi_x_test.err

module load miniconda  # or module load miniconda


# Activate the conda environment
conda activate cert  # or conda activate scoreenv if your cluster supports it directly

# Navigate to the directory with your script
cd /u/f_giacomarra/repos/Certified_Generation_4_Planning



python Models/score_based/exec_csdi_norm.py --model_name "OBS" --map_type "obs"  --nepochs 500 --config "base32.yaml"