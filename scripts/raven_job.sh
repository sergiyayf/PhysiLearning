#!/bin/bash -l

# Standard output and error:
#SBATCH -o ./logs/OutTrainJob.out.%j
#SBATCH -e ./logs/ErrTrainJob.err.%j
# Initial working directory:
#SBATCH -D ./../
# Job Name:
#SBATCH -J PhysiLearning
# Queue (Partition):
#SBATCH --partition=general
#SBATCH --mail-type=none
#SBATCH --mail-user=saif@mpl.mpg.de

module purge
module load gcc/11
module load anaconda/3/2023.03
# Export
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
#export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LD_LIBRARY_PATH=/u/saif/soft/libzmq/lib:$LD_LIBRARY_PATH
# for pinning threads correctly:
export OMP_PLACES=threads
# run a programm
srun --ntasks=1 --exclusive --cpus-per-task=72 --mem-per-cpu=200 python ./src/physilearning/train.py ${SLURM_JOBID}
