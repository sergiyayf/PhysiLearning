# imports
import os
import sys
from PC_environment import PC_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback 
import multiprocessing as mp
import yaml

def make_env(port, rank, job_name = '000000', config_file='config.yaml', seed=0):
        """
            Utility function for multiprocessed env.
            :param env_id: (str) the environment ID
            :param num_env: (int) the number of environments you wish to have in subprocesse
            :param seed: (int) the inital seed for RNG
            :param rank: (int) index of the subprocess
        """
        def _init():
            env = PC_env.from_yaml(config_file,port=str(port),job_name = job_name)
            env.seed(seed + rank)
            return env
        set_random_seed(seed)
        return _init

if __name__ == '__main__':
    config_file = 'config.yaml'
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
    n_steps = config['learning']['model']['n_steps']
    total_timesteps = config['learning']['model']['total_timesteps']
    verbose = config['learning']['model']['verbose']
    enable_loading = config['learning']['model']['load']['enable_loading']
    load_from_external_file = config['learning']['model']['load']['external_file_loading']
    external_file_name = config['learning']['model']['load']['external_file_name']

    # create callbacks 
    checkpoint_callback = CheckpointCallback(save_freq=save_freq,save_path = model_path, name_prefix = name_prefix)

    # create environment
    num_cpu = n_envs # This should be equal to the number of cores allocated by the job - (agent_buffer)
    
    # if single environment job to be run
    if n_envs == 1: 
        print('Training agent on one environment') 
        env = PC_env.from_yaml(config_file,port='0',job_name=sys.argv[1])
    else:
        print('Training agent on {0} environments'.format(num_cpu))
        # create the vectorized environment
        env = SubprocVecEnv([make_env(i,i,job_name = sys.argv[1],config_file=config_file) for i in range(num_cpu)])
    
    # create a model, either laod from pretrained one or initiate a new model
    if enable_loading == 1:
        if load_from_external_file == 1:
            # load from external file
            model = PPO.load(external_file_name, env=env, ent_coef=ent_coef, verbose=verbose, n_steps=n_steps)
        else:
            # find the odldest saved file
            # load
            newest_file = sorted([os.path.join('Training','Logs',f) for f in os.listdir('./Training/Logs/') ], key=os.path.getctime)[-1]

            model = PPO.load(oldest_file, env=env, ent_coef=ent_coef, verbose=verbose, n_steps=n_steps)
    else:
        model = PPO('MlpPolicy', env=env, tensorboard_log=log_path, ent_coef=ent_coef, verbose=verbose, n_steps=n_steps)
    
    # train model
    model.learn(total_timesteps=int(total_timesteps), callback=checkpoint_callback)
        
    env.close()
