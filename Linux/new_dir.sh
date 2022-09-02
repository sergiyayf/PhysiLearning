#!/bin/sh
mkdir $1
cp ./PhysiCell_V_1.10.4/*.o $1/
cp -r ./PhysiCell_V_1.10.4/custom_modules $1/
cp ./PhysiCell_V_1.10.4/run.sh $1/
cp -r ./PhysiCell_V_1.10.4/config $1/
cp ./PhysiCell_V_1.10.4/main.cpp $1/
cp ./PhysiCell_V_1.10.4/project $1/
cp -r ./PhysiCell_V_1.10.4/core $1/
cp ./PhysiCell_V_1.10.4/Makefile $1/
cp -r ./PhysiCell_V_1.10.4/output $1/
cp -r auxiliary $1/
