# PhysiLearning
[![CI](https://github.com/sergiyayf/PhysiLearning/actions/workflows/ci.yaml/badge.svg)](https://github.com/sergiyayf/PhysiLearning/actions/workflows/ci.yaml)
[![coverage](https://codecov.io/github/sergiyayf/PhysiLearning/branch/master/graph/badge.svg?token=EsiaxXIL7Z)](https://codecov.io/github/sergiyayf/PhysiLearning)
![version](https://img.shields.io/badge/version-0.1.1-blue)

PhysiLearning is a project in applying Reinforcement Learning to improve evolution based therapies
considering physical cell-cell interactions. This repository is mainly build on two great open source platforms:
PhysiCell - for simulating tumor growth, and Stable Baselines 3 - for reinforcement learning.

## Installation
Clone the repository and install the main package with pip
```bash
git clone
cd PhysiLearning
pip install -e .
```

You will also need to install ZMQ cpp library. On Ubuntu:
```bash
sudo apt-get install libzmq-dev
```

## Usage 

Usage of the package is aimed to be user-friendly and require minimal coding.
Most of the configuration is done through the config.yaml file that controls 
both the environment(simulation) and the agent.

See the config.yaml file for more details on the configuration, it should be self-explanatory.

### First steps 

To make sure that PhysiCell works on your machine, run the following command:
```bash
make raven
```
or 
```bash
make mela
```
depending on where you want to run the simulation. This will recompile PhysiCell with the options that are 
machine specific. 

### Training
To train the agent on ubuntu with installed slurm queuing system, run the following command:
```bash
python run.py train
```

### Evaluation
To evaluate the agent, run the following command:
```bash
python run.py evaluate
```

