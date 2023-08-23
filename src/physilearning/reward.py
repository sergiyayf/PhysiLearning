
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

        # increase reward for longer survival
        elif self.reward_shaping_flag == 5:
            if time_normalized < 0.5:
                reward = 1
            elif time_normalized < 0.75:
                reward = 2 
            elif time_normalized < 0.9:
                reward = 5
            else: 
                reward = 10

        else:
            raise ValueError("reward_shaping_flag not recognized")

        return reward
