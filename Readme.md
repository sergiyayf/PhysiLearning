# PhysiLearning
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/sergiyayf/PhysiLearning/ci.yaml)

PhysiLearning is a project on Adaptive Therapy (AT) using reinforcement learning and PhysiCell simulations.

More descriptive docs will come later.

Currently code is not cross-platform, and only works on Linux HPC and has a lot of dependecies: stable-baselines 3 and ZMQ. 
...

## Usage 

bin directory contains shell scripts for job submission, and some additional things to prep your job

Use 'single_env_job.sh' for training with one environment, and multiple_env.. for vector environment job. 

Thing will be added to the config.yaml file, so it contains all the necessary configurations for to train the agent. Very likely that PhysiCell simulations will still be configured with the PhysiCell_settings.xml . 
