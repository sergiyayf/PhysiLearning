#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./logs/Out_job_%j.out        # Standard output, %A = job ID, %a = job array index
#SBATCH -e ./logs/Err_job_%j.err        # Standard error, %A = job ID, %a = job array index
# Initial working directory:
#SBATCH -D ./../
# Job Name:
#SBATCH -J gen_tr
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2000MB

#SBATCH --mail-type=none
#SBATCH --mail-user=serhii.aif@mpl.mpg.de
#
# Wall clock limit (max. is 24 hours):
#SBATCH --time=24:00:00

# Load compiler and MPI modules (must be the same as used for compiling the code)
module purge
module load gcc/11
module load anaconda/3/plvenv
#module load anaconda/3/2021.05

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export LD_LIBRARY_PATH=/u/saif/soft/libzmq/lib:$LD_LIBRARY_PATH
export OMP_PLACES=threads
# Run the program:
#  the environment variable $SLURM_ARRAY_TASK_ID holds the index of the job array and
#  can be used to discriminate between individual elements of the job array
arg1=$1
arg2=$2

arg1=$($(arg1))
arg2=$($(arg2))
for ((i=$arg1; i<$arg2; i++)); do
    srun --ntasks=1 --exclusive --cpus-per-task=1 --mem-per-cpu=300  python3 ./scripts/simulate_patients.py --jobid=${SLURM_JOBID} --port=$i &
    srun --ntasks=1 --exclusive --cpus-per-task=1 --mem-per-cpu=300  python3 ./scripts/pcdl.py --jobid=${SLURM_JOBID} --port=$i &
done;

wait