#!/bin/bash -l
# specify the indexes (max. 30000) of the job array elements (max. 300 - the default job submit limit per user)
#SBATCH --array=1-100
# Standard output and error:
#SBATCH -o ./logs/Out_job_%A_%a.out        # Standard output, %A = job ID, %a = job array index
#SBATCH -e ./logs/Err_job_%A_%a.err        # Standard error, %A = job ID, %a = job array index
# Initial working directory:
#SBATCH -D ./../
# Job Name:
#SBATCH -J gen_tr
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2000MB
# for OpenMP:
#SBATCH --cpus-per-task=4
#
#SBATCH --mail-type=none
#SBATCH --mail-user=serhii.aif@mpl.mpg.de
#
# Wall clock limit (max. is 24 hours):
#SBATCH --time=24:00:00

# Load compiler and MPI modules (must be the same as used for compiling the code)
module purge
module load gcc/11
module load anaconda/3/2021.05

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LD_LIBRARY_PATH=/u/saif/soft/libzmq/lib:$LD_LIBRARY_PATH
export OMP_PLACES=cores
# Run the program:
#  the environment variable $SLURM_ARRAY_TASK_ID holds the index of the job array and
#  can be used to discriminate between individual elements of the job array

srun --cpus-per-task=4  python3 ./scripts/simulate_patients.py --jobid=${SLURM_JOBID} --taskid=${SLURM_ARRAY_TASK_ID}
