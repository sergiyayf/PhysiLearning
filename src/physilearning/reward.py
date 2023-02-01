# reward class
import numpy as np

class Reward:
    def __init__(self, reward_shaping_flag = 0):
        self.reward_shaping_flag = reward_shaping_flag

    def get_reward(self, obs):
        if self.reward_shaping_flag == 0:
            reward = 1
        elif self.reward_shaping_flag == 1:
            reward = 1 - obs[0]
        elif self.reward_shaping_flag == 2:
            reward = 1 - obs[1]
        elif self.reward_shaping_flag == 3:
            reward = 1 - obs[0] - obs[1]
        else:
            if np.sum(obs[0:2]) != 0:
                reward = 1 - obs[0] - 10 * obs[1]  # this means about 60 % of the simulation space is filled
            else:
                reward = 5

        return reward

