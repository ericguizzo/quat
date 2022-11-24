#!/bin/bash
#SBATCH -D /users/aczk407/archive/aczk407/eric_phd/sapienza/quat/src  # Working directory
#SBATCH --job-name=exp_quat                 # Job name
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=eric.guizzo@city.ac.uk         # Where to send mail	
#SBATCH --nodes=1                            # Run on 2 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=8                 # Use all the cores on each node
#SBATCH --mem=60G                              # Expected memory usage (0 means use all available memory)
#SBATCH --time=72:00:00                      # Time limit hrs:min:sec
#SBATCH --output=experiment_job_%j.out        # Standard output and error log [%j is replaced with the jobid]
#SBATCH --error=experiment_job_%j.error
#SBATCH --gres=gpu:1
#SBATCH --partition=gengpu

module load cuda/11.1
source ~/.bashrc
conda activate quat
python3 exp_instance_emodb_red.py