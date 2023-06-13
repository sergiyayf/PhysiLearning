#!/bin/bash

SCRIPT_PATH=$(dirname "$0")

cp -r ${SCRIPT_PATH}/../src/PhysiCell_src ${SCRIPT_PATH}/../simulations/PhysiCell_0
for ((i = 1; i <= ${1}; i++)); do 
	cp -r ${SCRIPT_PATH}/../simulations/PhysiCell_0 ${SCRIPT_PATH}/../simulations/PhysiCell_$i
		
done
 



