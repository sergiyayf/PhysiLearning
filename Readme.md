# PhysiLearning
[![CI](https://github.com/sergiyayf/PhysiLearning/actions/workflows/ci.yaml/badge.svg)](https://github.com/sergiyayf/PhysiLearning/actions/workflows/ci.yaml)
[![coverage](https://codecov.io/github/sergiyayf/PhysiLearning/branch/master/graph/badge.svg?token=EsiaxXIL7Z)](https://codecov.io/github/sergiyayf/PhysiLearning)
![version](https://img.shields.io/badge/version-0.1.6-blue)

PhysiLearning is a project in applying Reinforcement Learning to improve evolution based therapies
considering physical cell-cell interactions. This repository is mainly build on two great open source platforms:
[PhysiCell](https://github.com/MathCancer/PhysiCell) - for simulating tumor growth, and [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) - for reinforcement learning.

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
After you installed zmq update ZMQLIB flag in src/PhysiCell_src/Makefile with the path to the library.

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

#### List of example policies
This is for now only an example, these values will not work

| **Usage**       | **policy_kwargs**                                       | **Description**                                  |
|-----------------|---------------------------------------------------------|--------------------------------------------------|
| Number obs      | `dict('net_arch': dict('pi': [32, 32], 'vf': [32, 32))` | Control the size of actor and critic networks    |
| Image/Multi obs | `dict('cnn_output_dim': 16)`                            | Control number of extracted features from images |


### Evaluation
To evaluate the agent, run the following command:
```bash
python run.py evaluate
```

### Run tests 

To run all tests, run the following command:
```bash
make pytest
```
or run single tests with:
```bash
pytest tests/test_evaluate.py
```


## Changelog

#### 0.1.6 Major changes
- Fully migrateted environment construction to BaseEnv class
- Cleaned up config.yaml file, and moved most of the environment parameters outside of the specific environments
- Implemented dictionary observation space for multiobs training 
- Added policy_kwargs parameter to config.yaml file for changing the policy architecture for training 
- Created a list of example policies in the Readme file

#### 0.1.5 Major changes
- Added image observation for LV environment
- Changed trajectory attribute for image observation: now trajectory is always the number_trajectory, and image is in the image_trajectory
- Updated evaluation.py for new trajectory naming 
- Added new config parameter for LV environment: image_sampling_type - can be 'random' or 'dense' and creates
a proxy image of the tumor simulated by LV with either randomly placing cells on the grid, or placing them in a circle
#### 0.1.5 Minor changes
- Fixed stopping one time step too early for PC env with number observations