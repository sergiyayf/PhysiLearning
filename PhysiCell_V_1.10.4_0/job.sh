#!/bin/bash -l

# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J MechanoEvolution
# Queue (Partition):
#SBATCH --partition=general
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=3000
#for OpenMP:
#SBATCH --cpus-per-task=1
#
#SBATCH --mail-type=none
#SBATCH --mail-user=saif@mpl.mpg.de
#
# Wall clock limit:
#SBATCH --time=00:50:00

module purge
module load gcc/11
module load anaconda/3/2021.05
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# for pinning threads correctly:
export OMP_PLACES=cores
export LD_LIBRARY_PATH=/u/saif/soft/libzmq/lib:$LD_LIBRARY_PATH

#Run the program:
bash run.sh 
