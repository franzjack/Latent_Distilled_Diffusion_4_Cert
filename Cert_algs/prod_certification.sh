#!/bin/bash

#SBATCH -p Main
#SBATCH --job-name=Certify_x

#SBATCH -o run_cert_test.out
#SBATCH -e run_cert_test.err

module load miniconda  # or module load miniconda


# Activate the conda environment
conda activate cert  # or conda activate scoreenv if your cluster supports it directly

# Navigate to the directory with your script
cd /u/f_giacomarra/repos/Certified_Generation_4_Planning



python Cert_algs/certify_csdi.py 