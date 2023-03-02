#!/bin/bash -l

# Standard output and error:
#SBATCH -o ./logs/Outtjob.out.%j
#SBATCH -e ./logs/Errtjob.err.%j
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
module load anaconda/3/plvenv
# Export
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
#export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export LD_LIBRARY_PATH=/u/saif/soft/libzmq/lib:$LD_LIBRARY_PATH
# for pinning threads correctly:
#export OMP_PLACES=cores
# run a programm
srun python ./src/physilearning/training.py ${SLURM_JOBID}
