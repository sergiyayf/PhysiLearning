# imports
import os
from PC_environment import PC_env
from stable_baselines3 import PPO

# define paths
log_path = os.path.join('Training', 'Logs')
model_path = os.path.join('Training', 'SavedModels', 'PP0_at_1')


# create environment
env = PC_env()

# create model
model = PPO('MlpPolicy', env, tensorboard_log=log_path, ent_coef = 0.01, verbose = 1)

# train model
model.learn(total_timesteps=int(3))

# save model
model.save(model_path)

# evaluate model
#evaluate_policy(model, env, n_eval_episodes=5, render=False)
env.close()