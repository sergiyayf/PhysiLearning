import os
import sys
import yaml
import importlib
import multiprocessing as mp

from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.utils import set_random_seed

from physilearning.callbacks import CopyConfigCallback, SaveOnBestTrainingRewardCallback
from physilearning.envs.base_env import BaseEnv
import time

from typing import List, Callable, Optional, Dict, Any


def make_env(
    EnvClass: Callable = BaseEnv,
    *,
    config_file: str = 'config.yaml',
    env_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> Callable:
    """
        Utility function for env needed for wrappers.
        :param EnvClass: the environment class
        :param config_file: path to the config file.
        :param env_kwargs: keyword arguments to pass to the environment
        :param seed: seed for the environment
    """
    if env_kwargs is None:
        env_kwargs = {}

    if 'port' not in env_kwargs:
        port = 0
    else:
        port = int(env_kwargs['port'])

    def _init():
        env = EnvClass.from_yaml(config_file, **env_kwargs)
        # env.seed(seed+rank) # Seed in the env not implemented, This shouldn't be needed
        # as I am setting the random seed later
        return env
    if seed == 0:
        seed = int(time.time())
    set_random_seed(seed+port)
    return _init


class Trainer:
    """ Trainer class for reinforcement learning agents

    :param config_file: (str) path to the config file.

    """
    def __init__(self, config_file: str = 'config.yaml'):
        with open(config_file, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            print('Parsing config file {0}'.format(config_file))
        self.env = None
        self.config_file = config_file
        self.model = None

        # parse config file
        self.env_type = self.config['env']['type']
        self.policy = self.config['learning']['model']['policy']
        self.n_envs = self.config['env']['n_envs']
        self.wrap = self.config['env']['wrap']
        self.wrapper = self.config['env']['wrapper']
        self.wrapper_kwargs = self.config['env']['wrapper_kwargs']
        self.model_name = self.config['learning']['model']['name']
        self.enable_model_loading = self.config['learning']['model']['load']['enable_loading']
        self.model_load_last = self.config['learning']['model']['load']['last_model']
        self.model_kwargs = self.config['learning']['model']['model_kwargs']
        self.saved_model_name = self.config['learning']['model']['load']['saved_model_name']
        self.save_freq = self.config['learning']['model']['save_freq']
        self.model_save_prefix = self.config['learning']['model']['model_save_prefix']
        self.total_timesteps = self.config['learning']['model']['total_timesteps']

    def setup_env(self) -> None:
        """ Set up the environment for training

        Description
        -----------
        First check environment type, import the appropriate environment class, and create the environment object.
        Then check the number of environments to be used for training. If n_envs = 1, then create a single environment.
        If n_envs > 1, then create a vectorized environment.
        Wrapp the environment if specified in the config file.

        """
        # import environment class
        env_type = self.env_type
        EnvClass = getattr(importlib.import_module('physilearning.envs'), env_type)
        if env_type == 'PcEnv':
            env_kwargs = {'port': '0', 'job_name': sys.argv[1]}
        else:
            env_kwargs = {}

        # Single environment
        if self.n_envs == 1:
            print('Training on single environment')
            if self.wrap:
                if self.wrapper == 'VecFrameStack':
                    env = DummyVecEnv([make_env(EnvClass, env_kwargs=env_kwargs, config_file=self.config_file)])
                    self.env = VecFrameStack(env, **self.wrapper_kwargs)
                    self.env = VecMonitor(self.env, os.path.join('Training', 'Logs'))

                elif self.wrapper == 'DummyVecEnv':
                    self.env = DummyVecEnv([make_env(EnvClass, env_kwargs=env_kwargs, config_file=self.config_file)])
                    self.env = VecMonitor(self.env, os.path.join('Training', 'Logs'))
                else:
                    raise ValueError('Wrapper not recognized')
            else:
                self.env = EnvClass.from_yaml(self.config_file, **env_kwargs)
                self.env = Monitor(self.env, os.path.join('Training', 'Logs'))

        # VecEnv with n_envs > 1
        elif self.n_envs > 1:
            print('Training on {0} environments'.format(self.config['env']['n_envs']))
            if self.wrap:
                if self.wrapper == 'VecFrameStack':
                    env = DummyVecEnv([make_env(EnvClass, config_file=self.config_file,
                                       env_kwargs={'port': str(env_number), 'job_name': sys.argv[1]})
                                       for env_number in range(self.n_envs)])
                    self.env = VecFrameStack(env, **self.wrapper_kwargs)
                elif self.wrapper == 'DummyVecEnv':
                    self.env = DummyVecEnv([make_env(EnvClass, config_file=self.config_file,
                                            env_kwargs={'port': str(env_number), 'job_name': sys.argv[1]})
                                            for env_number in range(self.n_envs)])
                elif self.wrapper == 'SubprocVecEnv':
                    env = SubprocVecEnv([make_env(EnvClass, config_file=self.config_file,
                                         env_kwargs={'port': str(env_number), 'job_name': sys.argv[1]})
                                         for env_number in range(self.n_envs)])
                    self.env = VecFrameStack(env, **self.wrapper_kwargs)

                else:
                    raise ValueError('Wrapper not recognized')
            else:
                raise ValueError('Vector environment must be wrapped')
            self.env = VecMonitor(self.env, os.path.join('Training', 'Logs'))

    def setup_model(self) -> None:
        """ Set up the model for training"""
        # try to import model from stable_baselines3 first and then from sb3_contrib
        try:
            Algorithm = getattr(importlib.import_module('stable_baselines3'), self.model_name)
        except ModuleNotFoundError:
            print('Algorithm not found in stable_baselines3. Trying sb3_contrib...')
            try:
                Algorithm = getattr(importlib.import_module('sb3_contrib'), self.model_name)
            except ModuleNotFoundError:
                raise ValueError('Model not found in stable_baselines3 or sb3_contrib')
        else:
            print('Algorithm found in stable_baselines3. Using it...')

        # check if environment is set up
        if self.env is None:
            raise ValueError('Environment is not set up. Please, set up the environment before model!')
        else:
            # check if loading from external file
            if self.enable_model_loading:

                if self.model_load_last:
                    # get the last model in the directory
                    model_list = os.listdir(os.path.join('Training', 'SavedModels'))
                    model_list.sort()
                    print('Loading model ./Training/SavedModels/{0}'.format(model_list[-1]))
                    self.model = Algorithm.load(os.path.join('Training', 'SavedModels', model_list[-1]), env=self.env,
                                                tensorboard_log=os.path.join('Training', 'Logs'), **self.model_kwargs)

                else:
                    print('Loading model {0}'.format(self.saved_model_name))
                    self.model = Algorithm.load(self.saved_model_name, env=self.env,
                                                tensorboard_log=os.path.join('Training', 'Logs'), **self.model_kwargs)

            else:
                self.model = Algorithm(self.policy, env=self.env,
                                       tensorboard_log=os.path.join('Training', 'Logs'),
                                       **self.model_kwargs)

    def setup_callbacks(self) -> List:
        """ Set up checkpoints for training"""
        # Create the checkpoint callback
        if isinstance(self.save_freq, str):
            checkpoint_callback = \
                SaveOnBestTrainingRewardCallback(check_freq=self.model_kwargs['n_steps'],
                                                 log_dir=os.path.join('Training', 'Logs'),
                                                 save_dir=os.path.join('Training', 'SavedModels'),
                                                 save_name=self.model_save_prefix)
        else:
            checkpoint_callback = CheckpointCallback(save_freq=self.save_freq,
                                                     save_path=os.path.join('Training', 'SavedModels'),
                                                     name_prefix=self.model_save_prefix)
        # Create copy config callback
        copy_config_callback = CopyConfigCallback(self.config_file, self.model_save_prefix)

        return [checkpoint_callback, copy_config_callback]

    def learn(self) -> None:
        """ Train the model"""
        # Create the model
        self.setup_env()
        self.setup_model()

        # Collect callbacks
        callbacks = self.setup_callbacks()
        self.model.learn(total_timesteps=self.total_timesteps, callback=callbacks, tb_log_name=self.model_save_prefix)
        self.model.save(os.path.join('Training', 'SavedModels', self.model_save_prefix + '_final'))
        self.env.close()


def train() -> None:
    config_file = f'config_{sys.argv[1]}.yaml'
    # Create the training object
    trainer = Trainer(config_file)
    # Train the model
    trainer.learn()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    train()
