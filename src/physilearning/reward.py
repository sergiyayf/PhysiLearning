# reward class
import numpy as np

class Reward:
    def __init__(self, reward_shaping_flag = 0):
        self.reward_shaping_flag = reward_shaping_flag

    def get_reward(self, obs):
        # +1 for each time step
        if self.reward_shaping_flag == 0:
            reward = 1
        # 1 - number of sensitive cells
        elif self.reward_shaping_flag == 1:
            reward = 1 - obs[0]
        # 1 - number of resistant cells
        elif self.reward_shaping_flag == 2:
            reward = 1 - obs[1]
        # 1 - total number of cells
        elif self.reward_shaping_flag == 3:
            reward = 1 - obs[0] - obs[1]
        # not giving treatment reward
        elif self.reward_shaping_flag == 4:
            if obs[2] == 0:
                reward = 1
            else:
                reward = 0

        # 1 - number of sensitive cells - 10 * number of resistant cells
        else:
            if np.sum(obs[0:2]) != 0:
                reward = 1 - obs[0] - 10 * obs[1]  # this means about 60 % of the simulation space is filled
            else:
                reward = 5

        return reward

