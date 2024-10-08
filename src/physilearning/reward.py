import numpy as np
class Reward:
    def __init__(self, reward_shaping_flag=0, normalization=1):
        self.reward_shaping_flag = reward_shaping_flag
        self.normalization = normalization


    def get_reward(self, obs, time_normalized, threshold):
        # +1 for each time step
        if np.sum(obs[:2]) >= threshold and obs[2] == 0:
            reward = -20
        else:
            if self.reward_shaping_flag == 0:
                reward = 1

            # 1 - 0.5 * treatment
            elif self.reward_shaping_flag == 1:
                reward = 1 - 0.5*obs[2]
            # 1 - number of resistant cells
            elif self.reward_shaping_flag == 2:
                reward = 1 - 0.5*(obs[0]+obs[1])/self.normalization - 0.5*obs[2]
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

            # reward for decreasing probability of dyingr
            elif self.reward_shaping_flag == 6:
                reward = 1 - 0.5 * (obs[0] + obs[1])/self.normalization - 0.5 * obs[2]
            elif self.reward_shaping_flag == 7:
                reward = obs[0]
            elif self.reward_shaping_flag == 8:
                reward = 1 - 0.5 * obs[0]
            else:
                raise ValueError("reward_shaping_flag not recognized")
        return reward
    def tendayaverage(self, trajectory, time):
        r = trajectory[1,:]
        s = trajectory[0,:]
        tot = (r + s)/np.sum(trajectory[0:2,2])
        last_seven_days = tot[time-13:time+1]
        if np.any(last_seven_days < 1):
        #if np.sum(trajectory[0:2, time-19:time+1])/np.sum(trajectory[0:2,2]) < 20:
            reward = 1
        else:
            reward = 0
        return reward
