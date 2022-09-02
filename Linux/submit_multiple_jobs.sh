#!/bin/bash 
for ((i = 1; i <= ${2}; i++)); do 
	echo $i
	bash new_dir.sh "${1}_${i}"
	cd ${1}_${i}/auxiliary
	echo "dir = $(pwd)"
	sbatch job.sh
	cd ../..
done



