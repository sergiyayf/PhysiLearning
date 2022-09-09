#!/bin/bash -l

# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J PhysiLearning
# Queue (Partition):
#SBATCH --partition=general
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=33000MB
#for OpenMP:
#SBATCH --cpus-per-task=11
#
#SBATCH --mail-type=none
#SBATCH --mail-user=saif@mpl.mpg.de
#
# Wall clock limit:
#SBATCH --time=24:00:00

module purge
module load gcc/11
module load anaconda/3/2021.05


# Export
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LD_LIBRARY_PATH=/u/saif/soft/libzmq/lib:$LD_LIBRARY_PATH
# for pinning threads correctly:
export OMP_PLACES=cores

#NUM_RUNS=${SLURM_CPUS_PER_TASK}-1
#bash run.sh 0 &
#bash run.sh 1 &
#sleep 120 &
#wait
#for ((i=0; i<NUM_RUNS;i++)); do
#	srun --exclusive --ntasks=1 --cpus-per-task=1 --mem-per-cpu=3000 run.sh ${i} &
#done
#wait
#for ((i=0; i<NUM_RUNS;i++)); do
#	srun --exclusive --ntasks=1 --cpus-per-task=1 --mem-per-cpu=3000 run.sh ${i} &
#done
#wait
#Run the program:
#srun --exclusive --ntasks=1 --cpus-per-task=1 --mem-per-cpu=3000 run.sh 0  &
#srun --exclusive --ntasks=1 --cpus-per-task=1 --mem-per-cpu=3000 run.sh 1  &
#wait

srun --exclusive --ntasks=1 --cpus-per-task=1 --mem-per-cpu=3000 python3 ./vector_learning.py 
