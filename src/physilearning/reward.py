import numpy as np
class Reward:
    def __init__(self, reward_shaping_flag=0, normalization=1):
        self.reward_shaping_flag = reward_shaping_flag
        self.normalization = normalization

    def get_reward(self, obs, time, trajectory):

        if self.reward_shaping_flag == 'ttp':
            reward = 1
        elif self.reward_shaping_flag == 'dont_treat':
            if obs[2] == 0 and np.sum(obs[0:2]) < 1.5*np.sum(trajectory[0:2, 0]):
                reward = 1
            else:
                reward = 0
        elif self.reward_shaping_flag == 'seven_days_margin':
            r = trajectory[1, :]
            s = trajectory[0, :]
            tot = (r + s) / np.sum(trajectory[0:2, 0])
            last_seven_days = tot[time - 13:time + 1]
            if np.any(last_seven_days < 1):
                reward = 1
            else:
                reward = 0
        elif 'less_than' in self.reward_shaping_flag:
            cap = float(self.reward_shaping_flag[-3:])
            if np.sum(obs[0:2]) < cap:
                reward = 1
            else:
                reward = 0
        elif self.reward_shaping_flag == 'average':
            r = trajectory[1, :]
            s = trajectory[0, :]
            tot = (r + s) / np.sum(trajectory[0:2, 0])
            # average over last 10 days
            average = np.mean(tot[time - 20:time + 1])
            if average < 1.2:
                reward = 1
            else:
                reward = 0
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
