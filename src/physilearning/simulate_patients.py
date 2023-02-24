# imports
import os
from physilearning.PC_environment import PC_env
from physilearning.ODE_environments import LV_env
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import yaml
import click

def AT(obs, env, threshold=.60):
    """
    cycling adaptive therapy strategy
    """
    tumor_size = np.sum(obs[0:2])

    if tumor_size > threshold:
        action = 1
    else:
        action = 0
    return action


class Evaluation():

    def __init__(self, env):
        # Settign up evaluation
        # Create the environment
        self.env = env

    def therapy(self, obs, type='fixed', threshold=.50):
        tumor_size = np.sum(obs[0:2])
        if type == 'fixed':
            if tumor_size > threshold:
                action = 1
            else:
                action = 0
        elif type == 'random':
            action = np.random.binomial(1, threshold)
        return action

    def run(self, num_episodes=1, name='AT', run=1, path='./', type='fixed', threshold=.50):
        """ Run adaptive therapy for comperison"""
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            score = 0
            while not done:
                action = self.therapy(obs, type=type, threshold=threshold)
                # action = 1
                obs, reward, done, info = self.env.step(action)
                score += reward
            patientid = str(name)
            self.save_trajectory(type+str(threshold), patientid, run=run)

    def save_trajectory(self, type, patientid, run=1):
        """
        Save the trajectory to a csv file
        Parameters
        ----------
        episode

        Returns
        -------

        """
        h5File = 'simulated_patients.h5'
        df = pd.DataFrame(np.transpose(self.env.trajectory), columns=['Type 0', 'Type 1', 'Treatment'])
        df.to_hdf(h5File,f'data/{patientid}/{type}/run_{run}')
        return

@click.command()
@click.option('--jobid', default=0, help='ID of the job')
@click.option('--taskid', default=1, help='ID of the task')
def main(jobid, taskid):
    config_file = 'config.yaml'
    # do n Runs of different treatments
    n_full_treatment_runs = 2
    n_no_treatment_runs = 2
    n_adaptive_runs = 2
    n_random_runs = 2

    for i in range(0, n_full_treatment_runs):
        env = PC_env.from_yaml('config.yaml', port=str(taskid), job_name=str(jobid) + str(taskid) + str(i))
        evaluation = Evaluation(env)
        evaluation.run(num_episodes=1, name=f'patient_{jobid}_{taskid}', run=i, path='./', type='fixed', threshold=.0)
    for i in range(0, n_no_treatment_runs):
        env = PC_env.from_yaml('config.yaml', port=str(taskid), job_name=str(jobid) + str(taskid) + str(i))
        evaluation = Evaluation(env)
        evaluation.run(num_episodes=1, name=f'patient_{jobid}_{taskid}', run=i, path='./', type='fixed', threshold=1.)
    for i in range(0, n_adaptive_runs):
        env = PC_env.from_yaml('config.yaml', port=str(taskid), job_name=str(jobid) + str(taskid) + str(i))
        evaluation = Evaluation(env)
        evaluation.run(num_episodes=1, name=f'patient_{jobid}_{taskid}', run=i, path='./', type='fixed', threshold=.50)
    for i in range(0, n_random_runs):
        env = PC_env.from_yaml('config.yaml', port=str(taskid), job_name=str(jobid) + str(taskid) + str(i))
        evaluation = Evaluation(env)
        evaluation.run(num_episodes=1, name=f'patient_{jobid}_{taskid}', run=i, path='./', type='random', threshold=.05)
def test_job_array(jobid, taskid):
    for i in range(1, 3):
        print(f'jobid: {jobid}, taskid: {taskid}, i: {i}')
        config_file = 'config.yaml'
        env = PC_env.from_yaml(config_file, port=str(taskid), job_name=str(jobid) + str(taskid) + str(i))
        print('env_job_name: ', env.job_name)
        evaluation = Evaluation(env)
        evaluation.run_AT(num_episodes=1, name=f'AT_{jobid}_{taskid}_{i}', path='./')


if __name__ == '__main__':
    main()





