from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import matplotlib.pyplot as plt
from physilearning.reward import Reward

class jonaLVenv(Env):
    def __init__(self):

        # setup environment
        self.action_space = Discrete(2)  # Actions we can take: drug on or off
        self.observation_space = Box(low=0, high=100, shape=(3,))  # cancer burden and drug concentration

        # setup environment parameters
        self.time = 0  # initialize time
        self.max_time = 1000  # time to run simulation
        self.threshold_burden = 100  # threshold to stop simulation

        # setup initial variables
        self.initial_wt = 30
        self.initial_mut = 10
        self.initial_drug = 0
        self.burden = self.initial_wt + self.initial_mut


        # initialize state
        self.state = [self.initial_wt,
                      self.initial_mut,
                      self.initial_drug]
        # Set start burden and initial drug concentration

        # setup strain properties
        self.capacity = 1.5 * self.threshold_burden  # carrying capacity
        self.growth_rate = [0.015, 0.01]  # wt and mutant
        self.death_rate = [0.001, 0.001]  # wt and mutant
        self.death_rate_treat = [0.05, 0.001]
        self.competition = [3, 1]   # strength of competition (supressing growth of the other)

        # initialize trajectory recording
        self.trajectory = np.zeros((np.shape(self.state)[0],int(self.max_time)))

    def grow(self, i, j):  # i index of growing type, j: index of competing type
        new_pop_size = self.state[i] * \
                       (1 + self.growth_rate[i] *
                        (1 - (self.state[i] + self.state[j]*self.competition[j]) / self.capacity) -
                        self.death_rate[i] -
                        self.death_rate_treat[i]*self.state[2])

        return new_pop_size

    def step(self, action):

        # advance timer
        self.time += 1

        # grow wt
        self.state[0] = self.grow(0, 1)  # wt growth - drug effect
        self.state[0] = np.max([0, self.state[0]])  # ensure >0

        # grow mt
        self.state[1] = self.grow(1, 0)  # mt growth - drug effect
        self.state[1] = np.max([0, self.state[1]])  # ensure >0

        # update total burden (np.sum is exclusive)
        self.burden = np.sum(self.state[0:2])

        # Apply action
        # 0 drug off
        # 1 drug on
        self.state[2] = action

        # record trajectory
        self.trajectory[:, self.time - 1] = self.state

        # Calculate reward
        reward = 1    # reward long survival

        # Check if done
        if self.time >= self.max_time or self.burden <= 0 or self.burden >= self.threshold_burden:
            done = True

        else:
            done = False

        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, done, info

    def render(self, mode="human"):
        pass

    def reset(self):
        # Reset state
        self.state = [self.initial_wt, self.initial_mut, self.initial_drug]

        # Reset time
        self.time = 0

        # Reset trajectory
        self.trajectory = np.zeros((np.shape(self.state)[0], self.max_time))

        return self.state
