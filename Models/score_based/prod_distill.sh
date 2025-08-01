#!/bin/bash

#SBATCH -p lovelace
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --job-name=Distill_x

#SBATCH -o run_prod_distill_x_4.out
#SBATCH -e run_prod_distill_x_4.err

module load miniconda  # or module load miniconda


# Activate the conda environment
conda activate cert  # or conda activate scoreenv if your cluster supports it directly

# Navigate to the directory with your script
cd /u/f_giacomarra/repos/Certified_Generation_4_Planning



python Models/score_based/exec_distillation.py --model_name "OBS" --map_type "obs"  --nepochs 4000 --config "base32.yaml" --gamma 0.3