#!/bin/bash
#SBATCH -D /users/aczk407/test/quat/src  # Working directory
#SBATCH --job-name=test_1                 # Job name
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=eric.guizzo@city.ac.uk         # Where to send mail	
#SBATCH --exclusive                          # Exclusive use of nodes
#SBATCH --nodes=2                            # Run on 2 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=48                 # Use all the cores on each node
#SBATCH --mem=0                              # Expected memory usage (0 means use all available memory)
#SBATCH --time=72:00:00                      # Time limit hrs:min:sec
#SBATCH --output=myfluent_test_%j.out        # Standard output and error log [%j is replaced with the jobid]
#SBATCH --error=myfluent_test_%j.error

module load cuda/9.2

python3 dummy_job.py