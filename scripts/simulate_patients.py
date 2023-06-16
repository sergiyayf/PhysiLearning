# imports
from physilearning.envs import PcEnv
import numpy as np
import pandas as pd
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
        print(f'{tumor_size=}')
        print(f'{threshold=}')
        print(f'{self.env.state=}')
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
@click.option('--port', default=0, help='ID of the task')
def main(jobid, port):
    # do n Runs of different treatments

    n_no_treatment_runs = 1

    for i in range(0, n_no_treatment_runs):
        env = PcEnv.from_yaml('config.yaml', port=str(port), job_name=str(jobid) + str(i))
        evaluation = Evaluation(env)
        evaluation.run(num_episodes=1, name=f'patient_{jobid}_{port}', run=i, path='../src/physilearning/', type='fixed', threshold=1.)


if __name__ == '__main__':
    main()





