# reward class
import numpy as np

class Reward:
    def __init__(self, reward_shaping_flag=0, normalization=1):
        self.reward_shaping_flag = reward_shaping_flag
        self.normalization = normalization

    def get_reward(self, obs, time_normalized):
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

        # +1-normalized_cell_count-treatment_on
        elif self.reward_shaping_flag == 5:
            if np.sum(obs[0:2]) > 1e-6:
                reward = 1 - 0.1*np.sum(obs[0:2]) / self.normalization - 0.5*obs[2]
            else:
                reward = 10
        # increase reward for longer survival
        elif self.reward_shaping_flag == 6:
            if time_normalized < 0.5:
                reward = 1
            elif time_normalized < 0.75:
                reward = 2 
            elif time_normalized < 0.9:
                reward = 5
            elif time_normalized > 0.9:
                reward = 10 
            else: 
                reward = 0
        elif self.reward_shaping_flag == 7:
            if np.sum(obs) > 1e-5:
                reward = 0.1
            else:
                reward = 2

        # keep constant cell number
        elif self.reward_shaping_flag == 8:
            if abs(np.sum(obs)-500) < 50:
                reward = 1
            else:
                reward = 0
        # 1 - number of sensitive cells - 10 * number of resistant cells
        else:
            if np.sum(obs[0:2]) > 0:
                reward = 1 - obs[0] - 10 * obs[1]  # this means about 60 % of the simulation space is filled
            else:
                reward = 5

        return reward

