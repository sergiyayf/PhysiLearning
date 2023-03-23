import os
import sys
import yaml
import time
import importlib

from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from physilearning.callbacks import CopyConfigCallback


class Trainer():
    """ Trainer class for reinforcement learning agents

    :param config_file: (str) path to the config file.

    """
    def __init__(self, config_file: str = 'config.yaml')->None:
        with open(config_file, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            print('Parsing config file {0}'.format(config_file))
        self.env = None
        self.config_file = config_file
        self.model = None

    def setup_env(self) -> None:
        """ Set up the environment for training

        Description
        -----------
        First check environment type, import the appropriate environment class, and create the environment object.
        Then check the number of environments to be used for training. If n_envs = 1, then create a single environment.
        If n_envs > 1, then create a vectorized environment.
        Wrapp the environment if specified in the config file.

        """
        # Get environment type
        env_type = self.config['env']['type']
        if env_type == 'PhysiCell':
            from physilearning.envs.pc import PcEnv
            env = PcEnv.from_yaml(self.config_file, port='0', job_name=sys.argv[1])
        elif env_type == 'LV':
            from physilearning.envs.lv import LvEnv
            env = LvEnv.from_yaml(self.config_file, port='0', job_name=sys.argv[1])
        elif env_type == 'LatticeBased':
            from physilearning.envs.grid_env import GridEnv
            env = GridEnv.from_yaml(self.config_file)
        else:
            raise ValueError('Environment type not recognized')

        # Single environment
        if self.config['env']['n_envs'] == 1:
            print('Training on single environment')
            if self.config['env']['wrap']:
                if self.config['env']['wrapper'] == 'VecFrameStack':
                    env = make_vec_env(env, n_envs=1, seed=time.time(), vec_env_cls=DummyVecEnv, vec_env_kwargs=None)
                    self.env = VecFrameStack(env, **self.config['env']['wrapper_kwargs'])
                elif self.config['env']['wrapper'] == 'DummyVecEnv':
                    self.env = make_vec_env(env, n_envs=1, seed=time.time(), vec_env_cls=DummyVecEnv, vec_env_kwargs=self.config['env']['wrapper_kwargs'])
                else:
                    raise ValueError('Wrapper not recognized')
            else:
                self.env = env

        # VecEnv with n_envs > 1
        elif self.config['env']['n_envs'] > 1:
            print('Training on {0} environments'.format(self.config['env']['n_envs']))
            if self.config['env']['wrap']:
                if self.config['env']['wrapper'] == 'VecFrameStack':
                    env = make_vec_env(env, n_envs=self.config['env']['n_envs'], seed=time.time(), vec_env_cls=SubprocVecEnv, vec_env_kwargs=None)
                    self.env = VecFrameStack(env, self.config['env']['wrapper_kwargs'])
                elif self.config['env']['wrapper'] == 'DummyVecEnv':
                    self.env = make_vec_env(env, n_envs=self.config['env']['n_envs'], seed=time.time(), vec_env_cls=DummyVecEnv, vec_env_kwargs=self.config['env']['wrapper_kwargs'])
                elif self.config['env']['wrapper'] == 'SubprocVecEnv':
                    self.env = make_vec_env(env, n_envs=self.config['env']['n_envs'], seed=time.time(), vec_env_cls=SubprocVecEnv, vec_env_kwargs=self.config['env']['wrapper_kwargs'])
                else:
                    raise ValueError('Wrapper not recognized')
            else:
                raise ValueError('Vector environment must be wrapped')

    def setup_model(self):
        """ Set up the model for training"""
        # check if rPPO
        if self.config['learning']['model']['name'] == 'RecurrentPPO':
            Algorithm = getattr(importlib.import_module('sb3_contrib'),
                                self.config['learning']['model']['name'])
        else:
            Algorithm = getattr(importlib.import_module('stable_baselines3'), self.config['learning']['model']['name'])

        # check if environment is set up
        if self.env is None:
            raise ValueError('Environment is not set up. Please, set up the environment before model!')
        else:
            # check if loading from external file
            if self.config['learning']['model']['load']['enable_loading']:

                if self.config['learning']['model']['load']['last_model']:
                    # get the last model in the directory
                    model_list = os.listdir(os.path.join('Training', 'SavedModels'))
                    model_list.sort()
                    print('Loading model ./Training/SavedModels/{0}'.format(model_list[-1]))
                    self.model = Algorithm.load(os.path.join('Training', 'SavedModels', model_list[-1]),self.env, **self.config['learning']['model']['model_kwargs'])

                else:
                    print('Loading model {0}'.format(self.config['learning']['model']['load']['saved_model_name']))
                    self.model = Algorithm.load(self.config['learning']['model']['load']['saved_model_name'],self.env, **self.config['learning']['model']['model_kwargs'])

            else:
                self.model = Algorithm(self.config['learning']['model']['policy'], self.env, **self.config['learning']['model']['model_kwargs'])

    def setup_callbacks(self):
        """ Set up checkpoints for training"""
        # Create the checkpoint callback
        checkpoint_callback = CheckpointCallback(save_freq=self.config['learning']['model']['save_freq'], save_path=os.path.join('Training', 'SavedModels'), name_prefix=self.config['learning']['model']['model_save_prefix'])
        # Create copy config callback
        copy_config_callback = CopyConfigCallback(self.config_file, self.config['learning']['model']['model_save_prefix'])

        return [checkpoint_callback, copy_config_callback]

    def learn(self):
        """ Train the model"""
        # Create the model
        self.setup_env()
        self.setup_model()

        # Collect callbacks
        callbacks = self.setup_callbacks()
        self.model.learn(total_timesteps=self.config['learning']['model']['total_timesteps'], callback=callbacks, tb_log_name=self.config['learning']['model']['model_save_prefix'])
        self.model.save(os.path.join('Training', 'SavedModels', self.config['learning']['model']['model_save_prefix'] + '_final'))
        self.env.close()

def train():
    config_file = f'config_{sys.argv[1]}.yaml'
    # Create the training object
    trainer = Trainer(config_file)
    # Train the model
    trainer.learn()

if __name__ == '__main__':
    train()

