# imports
import yaml
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from physilearning.reward import Reward

# create environment
class LV_env(Env):
    def __init__(self, burden=1000, max_time=3000,
            initial_wt=45, initial_mut=5, treatment_time_step=60, reward_shaping_flag=0):
        # setting up environment
        # set up discrete action space
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0,high=1,shape=(3,))
        self.time = 0
        self.treatment_time_step = treatment_time_step
        self.max_time = max_time
        self.threshold_burden = burden

        self.initial_wt = initial_wt
        self.initial_mut = initial_mut
        self.initial_drug = 0
        self.burden = self.initial_mut+self.initial_wt
        self.state = [self.initial_wt/self.threshold_burden,
                      self.initial_mut/self.threshold_burden,
                      self.initial_drug]

        self.capacity = 1.5 * self.threshold_burden
        self.growth_rate = [0.0175,0.0175]
        self.death_rate = [0.001,0.001]
        self.current_death_rate = [0.001,0.001]
        self.death_rate_treat = [0.15,0.00]
        self.competition = [2.4e3,1]

        self.trajectory = np.zeros((np.shape(self.state)[0],self.max_time))
        self.real_step_count = 0#

        self.reward_shaping_flag = reward_shaping_flag

    @classmethod
    def from_yaml(cls, yaml_file, port='0', job_name='000000'):
        with open(yaml_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        burden = config['learning']['env']['burden']
        max_time = config['learning']['env']['max_time']
        initial_wt = config['learning']['env']['initial_wt']
        timestep = config['learning']['env']['treatment_time_step']
        initial_mut = config['learning']['env']['initial_mut']
        reward_shaping_flag = config['learning']['env']['reward_shaping']
        transport_type = config['global']['transport_type']
        transport_address = config['global']['transport_address']
        if transport_type == 'ipc://':
            transport_address = f'{transport_address}{job_name}{port}'
        else:
            warnings.warn('Transport type is different from ipc, please check the config file if everything is correct')
            transport_address = f'{transport_address}:{port}'

        return cls(burden=burden, max_time=max_time,
                   initial_wt=initial_wt, treatment_time_step=timestep, initial_mut=initial_mut,
                   reward_shaping_flag=reward_shaping_flag)
    def step(self, action):
        self.time += self.treatment_time_step
        # grow_tumor
        self.state[0] = self.grow(0,1)
        self.state[1] = self.grow(1,0)

        self.burden = np.sum(self.state[0:2])
        # do action (apply treatment or not)
        self.state[2] = action

        # record trajectory
        self.trajectory[:,self.time - 1] = self.state
        # get the reward
        rewards = Reward(self.reward_shaping_flag)
        reward = rewards.get_reward(self.state)
        """
        if self.reward_shaping_flag == 0:
            reward = 1
        elif self.reward_shaping_flag == 1:
            reward = 1 - self.state[0]
        elif self.reward_shaping_flag == 2:
            reward = 1 - self.state[1]
        elif self.reward_shaping_flag == 3:
            reward = 1 - self.state[0] - self.state[1]
        else:
            if np.sum(self.state[0:2]) != 0:
                reward = 1 - self.state[0] - 10 * self.state[
                    1]  # this means about 60 % of the simulation space is filled
            else:
                reward = 5
        """
        # check if we are done
        if self.time >= self.max_time or self.burden <= 0 or self.burden >= 1:
            done = True
        else:
            done = False
        info = {}


        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.real_step_count += 1

        self.state = [self.initial_wt/self.threshold_burden, self.initial_mut/self.threshold_burden, self.initial_drug]

        self.time = 0

        self.trajectory = np.zeros((np.shape(self.state)[0],self.max_time))
        self.current_death_rate = [self.death_rate[0],self.death_rate[1]]

        return self.state

    def grow(self, i, j):  # i index of growing type, j: index of competing type
        if self.state[2] == 0:
            if self.current_death_rate[0]>self.death_rate[0]:
                self.current_death_rate[0]*=0.80
            elif self.current_death_rate[0] < self.death_rate[0]:
                self.current_death_rate = [self.death_rate[0],self.death_rate[1]]

            new_pop_size = self.state[i] * \
                           (1 + self.growth_rate[i] * (1 - (self.state[i] + self.state[j] * self.competition[j]) / self.capacity) -
                            self.current_death_rate[i])
            new_pop_size = np.max([0,new_pop_size])
        else:
            self.current_death_rate[0] = 1.2*self.current_death_rate[0]
            new_pop_size = self.state[i] * \
                           (1 + self.growth_rate[i] * (1 - (self.state[i] + self.state[j] * self.competition[j]) / self.capacity) -
                            self.current_death_rate[i])
            new_pop_size = np.max([0,new_pop_size])
        return new_pop_size