#!/bin/bash 

SCRIPT_PATH=$(dirname "$0")
cd ${SCRIPT_PATH}/../simulations/PhysiCell_V_1.10.4_${1} && srun --exclusive --ntasks=1 --cpus-per-task=1 --mem-per-cpu=300 ./project ${2} &
#cd ${SCRIPT_PATH}/../simulations/PhysiCell_V_1.10.4_${1} && ./project ${2} &
