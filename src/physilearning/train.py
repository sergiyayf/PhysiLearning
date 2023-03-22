import os
import sys
import yaml
import time
import importlib

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO as rPPO
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
        self.env = self.setup_env()
        self.config_file = config_file
        self.policy = self.setup_policy()

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
                    self.env = VecFrameStack(env, **config['env']['wrapper_kwargs'])
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
                    self.env = VecFrameStack(env, config['env']['wrapper_kwargs'])
                elif self.config['env']['wrapper'] == 'DummyVecEnv':
                    self.env = make_vec_env(env, n_envs=self.config['env']['n_envs'], seed=time.time(), vec_env_cls=DummyVecEnv, vec_env_kwargs=config['env']['wrapper_kwargs'])
                elif self.config['env']['wrapper'] == 'SubprocVecEnv':
                    self.env = make_vec_env(env, n_envs=self.config['env']['n_envs'], seed=time.time(), vec_env_cls=SubprocVecEnv, vec_env_kwargs=config['env']['wrapper_kwargs'])
                else:
                    raise ValueError('Wrapper not recognized')
            else:
                raise ValueError('Vector environment must be wrapped')

    def setup_model(self):
        """ Set up the model for training"""
        Algorithm = getattr(importlib.import_module('stable_baselines3.ppo.policies'), self.config['learning']['model']['name'])

        # check if loading from external file
        if self.config['learning']['model']['load']['enable_loading']:

            if self.config['learning']['model']['load']['last_model']:
                # get the last model in the directory
                model_list = os.listdir(os.path.join('Training', 'SavedModels'))
                model_list.sort()
                print('Loading model ./Training/SavedModels/{0}'.format(model_list[-1])
                self.policy = Algorithm.load(os.path.join('Training', 'SavedModels', model_list[-1]),self.env, **self.config['learning']['model']['model_kwargs'])

            else:
                print('Loading model {0}'.format(self.config['learning']['model']['load']['saved_model_name']))
                self.policy = Algorithm.load(self.config['learning']['model']['load']['saved_model_name'],self.env, **self.config['learning']['model']['model_kwargs'])

        else:
            self.policy = Algorithm(self.config['learning']['model']['policy'], self.env, **self.config['learning']['model']['model_kwargs'])


if __name__ == '__main__':
    config_file = f'config_{sys.argv[1]}.yaml'
    with open(config_file,'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # define paths and laod others from config
    print('Parsing config file {0}'.format(config_file))
    log_path = os.path.join('Training', 'Logs')
    model_path = os.path.join('Training', 'SavedModels')
    save_freq = config['learning']['model']['save_freq']
    name_prefix = config['learning']['model']['model_save_prefix']
    n_envs = config['learning']['model']['n_envs']
    ent_coef = config['learning']['model']['ent_coef']
    clip_range = config['learning']['model']['clip_range']
    learning_rate = config['learning']['model']['learning_rate']
    n_steps = config['learning']['model']['n_steps']
    policy_kwargs = config['learning']['network']
    total_timesteps = config['learning']['model']['total_timesteps']
    verbose = config['learning']['model']['verbose']
    enable_loading = config['learning']['model']['load']['enable_loading']
    load_from_external_file = config['learning']['model']['load']['external_file_loading']
    external_file_name = config['learning']['model']['load']['external_file_name']
    env_type = config['env']['type']
    logname = config['learning']['model']['model_save_prefix']
    optimization_algorithm = config['learning']['model']['name']

    # create callbacks 
    checkpoint_callback = CheckpointCallback(save_freq=save_freq,save_path = model_path, name_prefix = name_prefix)
    copy_config_callback = CustomCallback(config_file,logname)
    #tensorboard_callback = TensorboardCallback()

    # create environment
    num_cpu = n_envs # This should be equal to the number of cores allocated by the job - (agent_buffer)
    
    # if single environment job to be run
    if n_envs == 1: 
        print('Training agent on one environment')
        if env_type == 'PhysiCell':
            from physilearning.envs.pc import PcEnv
            env = PcEnv.from_yaml(config_file,port='0',job_name=sys.argv[1])
        elif env_type == 'LV':
            from physilearning.envs.lv import LvEnv
            env = LvEnv.from_yaml(config_file,port='0',job_name=sys.argv[1])
        elif env_type == 'LatticeBased':
            from physilearning.envs.grid_env import GridEnv
            env = GridEnv()
            stack_frames = True
            if stack_frames:
                env = VecFrameStack(env, n_stack=4)
        else:
            raise ValueError('Environment type not recognized')

    else:
        print('Training agent on {0} environments'.format(num_cpu))
        # create the vectorized environment
        env = SubprocVecEnv([make_env(i,i,job_name = sys.argv[1],config_file=config_file) for i in range(num_cpu)])
    
    # create a model, either laod from pretrained one or initiate a new model
    if enable_loading == 1:
        if load_from_external_file == 1:
            # load from external file
            if optimization_algorithm == 'PPO':
                model = PPO.load(external_file_name, env=env, ent_coef=ent_coef, verbose=verbose, n_steps=n_steps,
                                 clip_range=clip_range, learning_rate=learning_rate)
            elif optimization_algorithm == 'RecurrentPPO':
                model = rPPO.load(external_file_name, env=env, ent_coef=ent_coef, verbose=verbose, n_steps=n_steps,
                                  clip_range=clip_range, learning_rate=learning_rate, policy_kwargs=policy_kwargs)
        else:
            # find the odldest saved file
            # load
            most_recent_file = \
            sorted([os.path.join('Training', 'SavedModels', f) for f in os.listdir('./Training/SavedModels/')],
                   key=os.path.getctime)[-1]
            #model_name = os.path.basename(most_recent_file).split('.')[0]

            model = PPO.load(most_recent_file, env=env, ent_coef=ent_coef, verbose=verbose, n_steps=n_steps,
                             clip_range=clip_range, learning_rate=learning_rate)
    else:
        if optimization_algorithm == 'PPO':
            print('Training agent with PPO algorithm')
            print(env.state)
            model = PPO('MlpPolicy', env=env, tensorboard_log=log_path, ent_coef=ent_coef, verbose=verbose,
                        n_steps=n_steps, clip_range=clip_range, learning_rate=learning_rate)
        elif optimization_algorithm == 'RecurrentPPO':
            model = rPPO('MlpLstmPolicy', env=env, tensorboard_log=log_path, ent_coef=ent_coef, verbose=verbose,
                         n_steps=n_steps, clip_range=clip_range, learning_rate=learning_rate, policy_kwargs=policy_kwargs)
        else:
            raise ValueError('Optimization algorithm not recognized')
    # train model

    model.learn(total_timesteps=int(total_timesteps), callback=[checkpoint_callback,copy_config_callback], tb_log_name=logname)
    model.save(os.path.join(model_path, logname+'_final'))
    env.close()
