#!/bin/bash


#SBATCH --job-name=sbi_cell_optimization     # Job name
#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=pbczgf@umsystem.edu    # Where to send mail	
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=1                   # Run a single task		
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=8gb                    # Job memory request
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output={$2}/cell_optimization_%j.log     # Standard output and error log
pwd; hostname; date

echo "Running sbi optimization on $SLURM_CPUS_ON_NODE CPU cores"

python optimize_cell.py $1 $2 

date