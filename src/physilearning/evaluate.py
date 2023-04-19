import os
from physilearning.envs import PcEnv, LvEnv, GridEnv
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import yaml
import warnings


def fixed_at(obs: np.ndarray, environment: LvEnv or PcEnv or GridEnv,
             threshold: float = .8, at_type: str = 'zhang_et_al') -> int:
    """ 
    Cycling adaptive therapy strategy, applies treatment only if the tumor burden is above a threshold

    :param obs: observation from the environment
    :param environment: environment
    :param threshold: threshold for the tumor burden
    :param at_type: type of adaptive therapy strategy
    :return: action
    """
    if environment.observation_space == 'image':
        warnings.warn('Fixed adaptive therapy for image observation space might not work as expected')
        
    tumor_size = np.sum(obs[0:2])
    if at_type == 'zhang_et_al':
        ini_tumor_size = environment.trajectory[0, 0] + environment.trajectory[1, 0]
        if tumor_size > ini_tumor_size:
            action = 1
        else:
            if env.trajectory[2, int(env.time) - 1] == 1 and tumor_size > threshold * ini_tumor_size:
                action = 1
            else:
                action = 0
    elif at_type == 'fixed':
        if tumor_size > threshold*environment.threshold_burden:
            action = 1
        else:
            action = 0
    elif at_type == 'mtd':
        action = 1
    else:
        action = 0

    return action 


class Evaluation():

    def __init__(self,env):
        # Settign up evaluation
        # Create the environment
        self.env = env


    def run_model(self, model_name, num_episodes = 1, path='./',name='model'):
        """
        Method to run the the environment with the loaded policy
        Parameters
        ----------
        num_episodes: int, number of episoded to run

        Returns
        -------

        """
        final_score = np.zeros(num_episodes) 
        model = PPO.load(model_name)
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
            filename = os.path.join(path, '{1}_{0}.csv'.format(name,episode))
            self.save_trajectory(filename)

        return
    
    def run_AT(self, num_episodes=1, name='AT', path='./'):
        """ Run adaptive therapy for comperison""" 
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False 
            score = 0
            while not done:
                #action = at_Zhang_et_al(obs,self.env)
                action = 0
                obs, reward, done, info = self.env.step(action)
                score += reward
            filename = os.path.join(path, '{1}_{0}.csv'.format(name,episode))
            if self.env.observation_type == 'image':
                np.save(f'trajectory_{name}_numpy', self.env.trajectory)
            else:
                self.save_trajectory(filename)

    def save_trajectory(self,name: str):
        """
        Save the trajectory to a csv file

        param: name: name of the file to save the trajectory

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
    #configure evaluation
    config_file = 'config.yaml'
    with open(config_file, 'r') as f:
        general_config = yaml.load(f, Loader=yaml.FullLoader)
        # define paths and load others from config
    print('Parsing config file {0}'.format(config_file))

    if general_config['eval']['from_file']:
        # configure environment and model to load
        model_training_path = general_config['eval']['path']
        model_prefix = general_config['eval']['model_prefix']
        model_config_file = os.path.join(model_training_path,'Training','Configs',model_prefix+'.yaml')
        with open(model_config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            # define paths and load others from config
        print('Parsing config file {0}'.format(model_config_file))

        #env_type = 'PhysiCell'
        env_type = general_config['eval']['evaluate_on']
        if env_type == 'PcEnv':
            env = PcEnv.from_yaml(model_config_file,port='0',job_name=sys.argv[1])
        elif env_type == 'LvEnv':
            env = LvEnv.from_yaml(model_config_file)
        elif env_type == 'GridEnv':
            env = GridEnv.from_yaml(model_config_file)
        else:
            raise Exception('Environment not recognized')

        evaluation = Evaluation(env)

        if general_config['eval']['fixed_AT_protocol']:
            evaluation.run_AT(num_episodes=general_config['eval']['num_episodes'], name=model_prefix, path=os.path.join(model_training_path,'Evaluations'))
        else:
            model_name = os.path.join(model_training_path, 'Training', 'SavedModels', model_prefix+general_config['eval']['step_to_load'])
            evaluation.run_model(model_name,num_episodes=general_config['eval']['num_episodes'],path=os.path.join(model_training_path,'Evaluations'),name=env_type+'Eval'+model_prefix)

    else:
        env_type = general_config['eval']['evaluate_on']
        save_name = general_config['eval']['save_name']
        if env_type == 'PcEnv':
            env = PcEnv.from_yaml(config_file,port='0',job_name=sys.argv[1])
        elif env_type == 'LV':
            env = LvEnv.from_yaml(config_file)
        elif env_type == 'GridEnv':
            env = GridEnv.from_yaml(config_file)
        else:
            raise Exception('Environment not recognized')

        evaluation = Evaluation(env)
        if general_config['eval']['fixed_AT_protocol']:
            evaluation.run_AT(num_episodes=general_config['eval']['num_episodes'],path='Evaluations', name=save_name)



    #most_recent_evaluation = 0
    #if most_recent_evaluation:

    #    most_recent_file = sorted([os.path.join('Training','SavedModels',f) for f in os.listdir('./Training/SavedModels/') ], key=os.path.getctime)[-1]
    #    model_name = os.path.basename(most_recent_file).split('.')[0]
    #    evaluation.run_model(model_name,num_episodes=3)
    #evaluation.run_model('./LV_not_treat_pretrained', num_episodes=1)
    #evaluation.run_model(r'060223_jonaEnv_test_rew+1_300000_steps', num_episodes=6)


    
