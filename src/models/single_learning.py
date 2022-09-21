# imports
import os
import sys
from PC_environment import PC_env
from stable_baselines3 import PPO
#from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

# define paths
log_path = os.path.join('Training', 'Logs')
model_path = os.path.join('Training', 'SavedModels')
checkpoint_callback = CheckpointCallback(save_freq=20000,save_path = model_path, name_prefix = 'PPO_vector')

if __name__ == '__main__':
    
    # create environment
    env = PC_env('0',job_name = sys.argv[1]) 
    
    model = PPO('MlpPolicy', env=env, tensorboard_log=log_path, ent_coef = 0.01, verbose = 1, n_steps = 300)
    
    # train model
    model.learn(total_timesteps=int(1e7), callback=checkpoint_callback)
    
    env.close()
