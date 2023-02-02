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

def AT(obs,env,threshold = 0.6):
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
        final_score = np.zeros(num_episodes) 
        model = PPO.load(os.path.join('Training', 'SavedModels', model_name))
        for episode in range(num_episodes):
            # reset the environment
            obs = self.env.reset()
            done = False
            score = 0

            while not done:
                action, _state = model.predict(obs)
                obs, reward, done, info = self.env.step(action)
                score += reward

            final_score[episode] = score
            print(f'Episode {episode} - Score: {score}')

            self.save_trajectory('eval_test_{0}.csv'.format(episode))

        return
    
    def run_AT(self, num_episodes=1):
        """ Run adaptive therapy for comperison""" 
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False 
            score = 0
            while not done:
                action = AT(obs,self.env)
                #action = 1
                obs, reward, done, info = self.env.step(action)
                score += reward
            
            self.save_trajectory('manual_AT_treatment_trajectory_{0}'.format(episode))

    def save_trajectory(self,name):
        """
        Save the trajectory to a csv file
        Parameters
        ----------
        episode

        Returns
        -------

        """
        df = pd.DataFrame(np.transpose(self.env.trajectory),columns=['Type 0', 'Type 1', 'Treatment'])
        df.to_csv(name)
        return

    def plot_trajectory(self, episode = 0):
        ax, fig = plt.subplots()
        df = pd.read_csv('trajectory_'+str(episode)+'.csv')
        df.plot(ax)

        return

if __name__ == '__main__':
    config_file = 'config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # define paths and load others from config
    print('Parsing config file {0}'.format(config_file))
    env_type = config['learning']['env']['type']
    if env_type == 'PhysiCell':
        env = PC_env.from_yaml(config_file,port='0',job_name=sys.argv[1])
    elif env_type == 'LV':
        env = LV_env.from_yaml(config_file)

    evaluation = Evaluation(env)
    most_recent_evaluation = 1
    if most_recent_evaluation: 

        most_recent_file = sorted([os.path.join('Training','SavedModels',f) for f in os.listdir('./Training/SavedModels/') ], key=os.path.getctime)[-1]
        model_name = os.path.basename(most_recent_file).split('.')[0]
        evaluation.run_model(model_name,num_episodes=3)
    #evaluation.run_model('./LV_not_treat_pretrained', num_episodes=1)
    #evaluation.run_AT(num_episodes=1)
    
