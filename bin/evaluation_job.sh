#!/bin/bash -l

# Standard output and error:
#SBATCH -o ./logs/tjob.out.%j
#SBATCH -e ./logs/tjob.err.%j
# Initial working directory:
#SBATCH -D ./../
# Job Name:
#SBATCH -J PhysiLearning
# Queue (Partition):
#SBATCH --partition=general
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=6000MB
#for OpenMP:
#SBATCH --cpus-per-task=2
#
#SBATCH --mail-type=none
#SBATCH --mail-user=saif@mpl.mpg.de
#
# Wall clock limit:
#SBATCH --time=01:00:00

module purge 
module load anaconda/3/plvenv
# Export
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# for pinning threads correctly:
export OMP_PLACES=cores

srun --ntasks=1 --cpus-per-task=1 --mem-per-cpu=3000 python3 ./src/physilearning/evaluation.py ${SLURM_JOBID}