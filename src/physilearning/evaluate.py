import os
from physilearning.envs import PcEnv, LvEnv, GridEnv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import warnings
import importlib
from typing import Dict, Optional, Any


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
        ini_tumor_size = environment.initial_wt + environment.initial_mut
        if tumor_size > ini_tumor_size:
            action = 1
        else:
            warnings.warn('This implementation is sensitive to the type of observation space, be careful')
            if environment.trajectory[2, int(environment.time) - 1] == 1 and tumor_size > threshold * ini_tumor_size:
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


class Evaluation:

    def __init__(self, env: LvEnv or PcEnv or GridEnv, config_file: str = 'config.yaml') -> None:
        """
        Class to evaluate the trained model

        :param env: environment
        :param config_file: configuration file
        """

        self.env = env
        with open(config_file, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def run_environment(
        self,
        model_name: str = '',
        num_episodes: int = 1,
        save_path: str = './',
        save_name: str = 'model',
        fixed_therapy: bool = True,
        fixed_therapy_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Run the environment on the trained policy.

        :param model_name: name of the model to load
        :param num_episodes: number of episodes to run
        :param save_path: path to save the trajectory
        :param save_name: name of the file to save the trajectory
        :param fixed_therapy: if true, run fixed adaptive therapy
        :param fixed_therapy_kwargs: keyword arguments for fixed adaptive therapy

        """
        if not fixed_therapy:
            algorithm_name = self.config['learning']['model']['name']
            try:
                Algorithm = getattr(importlib.import_module('stable_baselines3'), algorithm_name)
            except ModuleNotFoundError:
                print('Algorithm not found in stable_baselines3. Trying sb3_contrib...')
                try:
                    Algorithm = getattr(importlib.import_module('sb3_contrib'), algorithm_name)
                except ModuleNotFoundError:
                    raise ValueError('Model not found in stable_baselines3 or sb3_contrib')
            else:
                print('Algorithm found in stable_baselines3. Using it...')

            final_score = np.zeros(num_episodes)
            model = Algorithm.load(model_name)

        else:
            final_score = np.zeros(num_episodes)
            model = None

        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            score = 0

            while not done:
                if fixed_therapy:
                    action = fixed_at(obs, self.env, **fixed_therapy_kwargs)
                else:
                    action, _state = model.predict(obs)
                obs, reward, done, info = self.env.step(action)
                score += reward

            final_score[episode] = score
            print(f'Episode {episode} - Score: {score}')
            filename = os.path.join(save_path, '{1}_{0}.csv'.format(save_name, episode))
            self.save_trajectory(filename)

        return

    def save_trajectory(self, save_name: str) -> None:
        """
        Save the trajectory to a csv file

        param: save_name: name of the file to save the trajectory
        """
        if self.env.observation_type == 'image':
            np.save(f'trajectory_{save_name}_numpy', self.env.trajectory)
        else:
            df = pd.DataFrame(np.transpose(self.env.trajectory), columns=['Type 0', 'Type 1', 'Treatment'])
            df.to_csv(save_name)
        return None

    def plot_trajectory(self, episode: int = 0) -> None:
        """
        Plot the trajectory

        :param episode: episode to plot
        """
        ax, fig = plt.subplots()
        df = pd.read_csv('trajectory_'+str(episode)+'.csv')
        df.plot(ax)

        return


def evaluate() -> None:
    """
    Evaluate the trained model based on the evaluation specs in the config file
    """

    # configure evaluation
    config_file = 'config.yaml'
    with open(config_file, 'r') as f:
        general_config = yaml.load(f, Loader=yaml.FullLoader)
        # define paths and load others from config
    print('Parsing config file {0}'.format(config_file))

    if general_config['eval']['from_file']:
        # configure environment and model to load
        model_training_path = general_config['eval']['path']
        model_prefix = general_config['eval']['model_prefix']
        model_config_file = os.path.join(model_training_path, 'Training', 'Configs', model_prefix + '.yaml')

        env_type = general_config['eval']['evaluate_on']
        EnvClass = getattr(importlib.import_module('physilearning.envs'), env_type)
        env = EnvClass.from_yaml(model_config_file)
        evaluation = Evaluation(env)

        model_name = os.path.join(model_training_path, 'Training', 'SavedModels',
                                  model_prefix + general_config['eval']['step_to_load'])
        fixed = general_config['eval']['fixed_AT_protocol']
        evaluation.run_environment(model_name, num_episodes=general_config['eval']['num_episodes'],
                                   save_path=os.path.join(model_training_path, 'Evaluations'),
                                   save_name=env_type + 'Eval' + model_prefix, fixed_therapy=fixed)

    else:
        env_type = general_config['eval']['evaluate_on']
        save_name = general_config['eval']['save_name']
        EnvClass = getattr(importlib.import_module('physilearning.envs'), env_type)
        env = EnvClass.from_yaml(config_file)
        evaluation = Evaluation(env)

        fixed = general_config['eval']['fixed_AT_protocol']
        evaluation.run_environment(model_name='None', num_episodes=general_config['eval']['num_episodes'],
                                   save_path=os.path.join('.', 'Evaluations'),
                                   save_name=env_type+'Eval'+save_name, fixed_therapy=fixed)
    return


if __name__ == '__main__':
    evaluate()
