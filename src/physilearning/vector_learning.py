# imports
import os
import sys
from PC_environment import PC_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback 
import multiprocessing as mp
# define paths
log_path = os.path.join('Training', 'Logs')
model_path = os.path.join('Training', 'SavedModels')
checkpoint_callback = CheckpointCallback(save_freq=20000,save_path = model_path, name_prefix = 'PPO_vector')

def make_env(port, rank, job_name = '000000', seed=0):
        """
            Utility function for multiprocessed env.
            :param env_id: (str) the environment ID
            :param num_env: (int) the number of environments you wish to have in subprocesse
            :param seed: (int) the inital seed for RNG
            :param rank: (int) index of the subprocess
        """
        def _init():
            env = PC_env(str(port),job_name = job_name)
            env.seed(seed + rank)
            return env
        set_random_seed(seed)
        return _init

if __name__ == '__main__':
    mp.set_start_method('spawn')
    # create environment
    num_cpu = 10 # Number of cores
    # create the vectorized environment
    env = SubprocVecEnv([make_env(i,i,job_name = sys.argv[1]) for i in range(num_cpu)])
    
    model = PPO('MlpPolicy', env, tensorboard_log=log_path, ent_coef = 0.01, verbose = 1, n_steps = 300)
    
    # train model
    model.learn(total_timesteps=int(1e7), callback=checkpoint_callback)
    
    # evaluate model
    #evaluate_policy(model, env, n_eval_episodes=5, render=False)
    env.close()
