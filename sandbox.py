# evaluation of the LatticeBased run

from physilearning.envs.grid_env import GridEnv

from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import yaml

# setup environment
env = GridEnv.from_yaml(r'.\Training\Configs\LatticeBased_23_03_CNN_1_env_not_stacked.yaml')

# load pretrained model
model = PPO.load(r'.\data\2403_grid_mela_models\LatticeBased_24_03_CNN_1_env_not_stacked_mela_final', env=env)

done = False
obs = env.reset()

rew = 0
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    rew+=reward

print('reward: ', rew)
anim = env.render()
plt.show()
