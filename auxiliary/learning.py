# imports
import os
from PC_environment import PC_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback 
import subprocess

# define paths
log_path = os.path.join('Training', 'Logs')
model_path = os.path.join('Training', 'SavedModels')
checkpoint_callback = CheckpointCallback(save_freq=5000,save_path = model_path, name_prefix = 'PPO_burden')

# create environment
env = PC_env('1')

# create model
model = PPO('MlpPolicy', env, tensorboard_log=log_path, ent_coef = 0.01, verbose = 1)

# start the simulation 
# train model
model.learn(total_timesteps=int(2e5),callback=checkpoint_callback)

# save model
#model.save(model_path)
# evaluate model
#evaluate_policy(model, env, n_eval_episodes=5, render=False)
env.close()
