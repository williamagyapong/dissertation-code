#!/bin/bash
#SBATCH --job-name=BIPF_SiM
#SBATCH -p general
##SBATCH -A awagler2
#SBATCH -n 1                        # Number of nodes, constrained by the parallelization procedure
#SBATCH --ntasks=1                  # Total number of MPI tasks (1 per core)
#SBATCH --cpus-per-task=36          # Cores per task (1 per MPI process)
##SBATCH --time=03:00:00             # Wall time (adjust as needed)
#SBATCH --mem=0                     # Use all memory on allocated node
#SBATCH -o sim_out.txt
#SBATCH -e sim_err.txt

# Load python if needed
# module load python/3.x

# Activate virtual environment (if applicable)
 source $HOME/WilliamDissertation/bipfenv/bin/activate



# Run your simulation
python3 sim-studies-parallelized.py
