# imports
import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np

# create environment
class AT_env(Env):
    def __init__(self):
        # setting up environment
        # set up discrete action space
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0,high=1,shape=(3,))

        self.time = 0

        self.max_time = 500
        self.threshold_burden = 1000

        self.initial_wt = 45
        self.initial_mut = 5
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
        self.real_step_count = 0

    def step(self, action):
        self.time += 1
        # grow_tumor
        self.state[0] = self.grow(0,1)
        self.state[1] = self.grow(1,0)

        self.burden = np.sum(self.state[0:2])
        # do action (apply treatment or not)
        self.state[2] = action

        # record trajectory
        self.trajectory[:,self.time - 1] = self.state
        # get the reward
        if self.state[2] == 0:
            reward = 1
        else:
            reward = 0
        # reward = self.state[2]
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