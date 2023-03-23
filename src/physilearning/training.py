# imports
import os
import sys
import warnings

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO as rPPO
from physilearning.callbacks import CustomCallback
import yaml
from stable_baselines3.common.env_util import make_vec_env

def make_env(port, rank, job_name = '000000', config_file='config.yaml', seed=0):
        """
            Utility function for multiprocessed env.
            :param env_id: (str) the environment ID
            :param num_env: (int) the number of environments you wish to have in subprocesse
            :param seed: (int) the inital seed for RNG
            :param rank: (int) index of the subprocess
        """
        from physilearning.envs.pc import PcEnv
        def _init():
            env = GridEnv()
            env.seed(seed + rank)
            return env
        set_random_seed(seed)
        return _init

class Trainer():
    """ Trainer class for reinforcement learning agents
    Returns
    -------

    """
    def __init__(self, config_file: str = 'config.yaml')->None:
        with open(config_file, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def setup_env(self):
        """ Setup the environment for training
        Option 1 - Single environment
        Option 2 - Dummy Vectorized environment
        Option 3 - Subprocess Vectorized environment
        Option 4 - Stacked Vectorized environment

        Returns
        -------
        env: environment object to train the agent on
        """
        # option 1 - single environment
        if self.config['learning']['model']['n_envs'] == 1:
            print('Training on single environment')
            if env_type == 'PhysiCell':
                from physilearning.envs.pc import PcEnv
                env = PcEnv.from_yaml(config_file, port='0', job_name=sys.argv[1])
            elif env_type == 'LV':
                from physilearning.envs.lv import LvEnv
                env = LvEnv.from_yaml(config_file, port='0', job_name=sys.argv[1])
            elif env_type == 'LatticeBased':
                from physilearning.envs.grid_env import GridEnv
                env = GridEnv.from_yaml(config_file)
            else:
                raise ValueError('Environment type not recognized')

        return env



if __name__ == '__main__':
    warnings.warn('This script is deprecated. Please use the train.py script instead', DeprecationWarning)
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
            env = make_vec_env(GridEnv, n_envs=1)
            print('hi there')
            print(env.num_envs)
            stack_frames = True
            if stack_frames:
                env = VecFrameStack(env, n_stack=4, channels_order='last')
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

            model = PPO('MlpPolicy', env=env, tensorboard_log=log_path, ent_coef=ent_coef, verbose=verbose,
                        n_steps=n_steps, clip_range=clip_range, learning_rate=learning_rate)
        elif optimization_algorithm == 'RecurrentPPO':
            model = rPPO('MlpLstmPolicy', env=env, tensorboard_log=log_path, ent_coef=ent_coef, verbose=verbose,
                         n_steps=n_steps, clip_range=clip_range, learning_rate=learning_rate, policy_kwargs=policy_kwargs)
        else:
            raise ValueError('Optimization algorithm not recognized')
    # train model

    if env_type == 'LatticeBased':
        model = PPO('CnnPolicy', env=env, tensorboard_log=log_path, ent_coef=ent_coef, verbose=verbose,
                    n_steps=n_steps, clip_range=clip_range, learning_rate=learning_rate)
    model.learn(total_timesteps=int(total_timesteps), callback=[checkpoint_callback,copy_config_callback], tb_log_name=logname)
    model.save(os.path.join(model_path, logname+'_final'))
    env.close()
