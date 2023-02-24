#!/bin/bash -l
# specify the indexes (max. 30000) of the job array elements (max. 300 - the default job submit limit per user)
#SBATCH --array=1-10
# Standard output and error:
#SBATCH -o ./logs/Out_job_%A_%a.out        # Standard output, %A = job ID, %a = job array index
#SBATCH -e ./logs/Err_job_%A_%a.err        # Standard error, %A = job ID, %a = job array index
# Initial working directory:
#SBATCH -D ./../
# Job Name:
#SBATCH -J test_array
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60MB
# for OpenMP:
#SBATCH --cpus-per-task=1
#
#SBATCH --mail-type=none
#SBATCH --mail-user=serhii.aif@mpl.mpg.de
#
# Wall clock limit (max. is 24 hours):
#SBATCH --time=01:00:00

# Load compiler and MPI modules (must be the same as used for compiling the code)
module purge
module load anaconda/3/plvenv

# Run the program:
#  the environment variable $SLURM_ARRAY_TASK_ID holds the index of the job array and
#  can be used to discriminate between individual elements of the job array

srun --exclusive --cpus-per-task=10  python3 ./src/physilearning/simulate_patients.py --jobid=${SLURM_JOBID} --taskid=${SLURM_ARRAY_TASK_ID}