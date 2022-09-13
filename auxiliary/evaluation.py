# imports
import os
from PC_environment import PC_env
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Evaluation():

    def __init__(self,env):
        # Settign up evaluation
        # Create the environment
        self.env = env


    def run_model(self, model_name, num_episodes = 1):
        """
        Method to run the the environment with the loaded policy
        Parameters
        ----------
        num_episodes: int, number of episoded to run

        Returns
        -------

        """
        model = PPO.load(os.path.join('Training', 'SavedModels', model_name))
        for episode in range(num_episodes):
            # reset the environment
            obs = self.env.reset()
            done = False
            score = 0

            while not done:
                action, _state = model.predict(obs)
                obs, reward, done, info = env.step(action)
                score += reward

            final_score[episode] = score
            print(f'Episode {episode} - Score: {score}')

            self.save_trajectory(episode)

        return

    def save_trajectory(self,episode):
        """
        Save the trajectory to a csv file
        Parameters
        ----------
        episode

        Returns
        -------

        """
        df = pd.DataFrame(self.env.trajectory,columns=['Type 0', 'Type 1', 'Treatment'])
        df.to_csv('trajectory_'+str(episode)+'.csv')
        return

    def plot_trajectory(self, episode = 0):
        ax, fig = plt.subplots()
        df = pd.read_csv('trajectory_'+str(episode)+'.csv')
        df.plot(ax)

        return

if __name__ == '__main__':
    env = PC_env()
    eval = Evaluation(env)