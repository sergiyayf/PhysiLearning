#!/bin/bash -l

# Standard output and error:
#SBATCH -o ./logs/OutEvaltjob.out.%j
#SBATCH -e ./logs/ErrEvaltjob.err.%j
# Initial working directory:
#SBATCH -D ./../
# Job Name:
#SBATCH -J Evaluation
# Queue (Partition):
#SBATCH --partition=general
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=3000MB
#for OpenMP:
#SBATCH --cpus-per-task=14
#
#SBATCH --mail-type=none
#SBATCH --mail-user=saif@mpl.mpg.de
#
# Wall clock limit:
#SBATCH --time=10:00:00

module purge 
module load anaconda/3/plvenv
# Export
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# for pinning threads correctly:
export OMP_PLACES=cores

srun --ntasks=1 --cpus-per-task=1 --mem-per-cpu=500 python3 ./src/physilearning/evaluate.py ${SLURM_JOBID}
