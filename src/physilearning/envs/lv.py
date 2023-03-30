import yaml
from gym.spaces import Discrete, Box
import numpy as np
from physilearning.envs.base_env import BaseEnv
from physilearning.reward import Reward
from typing import Optional, Tuple, Union


class LvEnv(BaseEnv):
    """
    Environment for Lottka-Volterra tumor growth model

    :param max_tumor_size: (float) maximum tumor size in number of cells
    :param max_time: (int) maximum time in days
    :param carrying_capacity: (float) carrying capacity of the tumor
    :param initial_wt: (float) initial number of wild type cells
    :param initial_mut: (float) initial number of mutant cells
    :param growth_rate_wt: (float) growth rate of wild type cells
    :param growth_rate_mut: (float) growth rate of mutant cells
    :param death_rate_wt: (float) death rate of wild type cells
    :param death_rate_mut: (float) death rate of mutant cells
    :param treat_death_rate_wt: (float) death rate of wild type cells under treatment
    :param treat_death_rate_mut: (float) death rate of mutant cells under treatment
    :param competition_wt: (float) competition coefficient of wild type cells
    :param competition_mut: (float) competition coefficient of mutant cells
    :param treatment_time_step: (int) time step for treatment
    :param reward_shaping_flag: (int) flag for reward shaping
    :param growth_function_flag: (int) flag for growth function
    :param normalize: (bool) flag for normalization
    :param normalize_to: (float) normalization factor for reward
    """

    def __init__(
        self,
        max_tumor_size: float = 1000,
        max_time: int = 3000,
        carrying_capacity: float = 1500,
        initial_wt: float = 45,
        initial_mut: float = 5,
        growth_rate_wt: float = 0.0175,
        growth_rate_mut: float = 0.0175,
        death_rate_wt: float = 0.001,
        death_rate_mut: float = 0.001,
        treat_death_rate_wt: float = 0.15,
        treat_death_rate_mut: float = 0.0,
        competition_wt: float = 2.4e3,
        competition_mut: float = 1.0,
        treatment_time_step: int = 60,
        reward_shaping_flag: int = 0,
        growth_function_flag: int = 0,
        normalize:  bool = 1,
        normalize_to: float = 1000
    ) -> None:
        # Spaces
        self.name = 'LvEnv'
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0,high=normalize_to,shape=(1,))

        # Parameters
        self.time = 0
        self.treatment_time_step = treatment_time_step
        self.max_time = max_time
        self.threshold_burden_in_number = max_tumor_size
        self.threshold_burden = normalize_to
        self.wt_random = isinstance(initial_wt, str)
        if self.wt_random:
            self.initial_wt = np.random.random_integers(low=0, high=self.threshold_burden_in_number, size=1)[0]
            self.initial_wt = self.initial_wt*self.threshold_burden/self.threshold_burden_in_number
        else:
            self.initial_wt = initial_wt*normalize_to/max_tumor_size
        self.mut_random = isinstance(initial_mut, str)
        if self.mut_random:
            self.initial_mut = np.random.random_integers(low=0, high=0.01*self.threshold_burden_in_number, size=1)[0]
            self.initial_mut = self.initial_mut*self.threshold_burden/self.threshold_burden_in_number
        else:
            self.initial_mut = initial_mut*normalize_to/max_tumor_size
        self.initial_drug = 0
        self.burden = self.initial_mut+self.initial_wt
        self.state = [self.initial_wt,
                      self.initial_mut,
                      self.initial_drug]
        self.capacity = carrying_capacity*normalize_to/max_tumor_size
        # 1 - wt, 2 - resistant
        self.growth_rate = [growth_rate_wt,growth_rate_mut]
        self.death_rate = [death_rate_wt,death_rate_mut]
        self.current_death_rate = [death_rate_wt,death_rate_mut]
        self.death_rate_treat = [treat_death_rate_wt,treat_death_rate_mut]
        self.competition = [competition_wt,competition_mut]
        self.growth_function_flag = growth_function_flag

        self.trajectory = np.zeros((np.shape(self.state)[0],int(self.max_time)))
        self.real_step_count = 0

        self.reward_shaping_flag = reward_shaping_flag

    @classmethod
    def from_yaml(cls, yaml_file: str):
        with open(yaml_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return cls(max_tumor_size=config['env']['threshold_burden'],
                   max_time=config['env']['max_time'],
                   initial_wt=config['env']['LV']['initial_wt'],
                   treatment_time_step=config['env']['treatment_time_step'],
                   initial_mut=config['env']['LV']['initial_mut'],
                   reward_shaping_flag=config['env']['reward_shaping'],
                   carrying_capacity=config['env']['LV']['carrying_capacity'],
                   growth_rate_wt=config['env']['LV']['growth_rate_wt'],
                   growth_rate_mut=config['env']['LV']['growth_rate_mut'],
                   death_rate_wt=config['env']['LV']['death_rate_wt'],
                   death_rate_mut=config['env']['LV']['death_rate_mut'],
                   competition_wt=config['env']['LV']['competition_wt'],
                   competition_mut=config['env']['LV']['competition_mut'],
                   treat_death_rate_wt=config['env']['LV']['treat_death_rate_wt'],
                   treat_death_rate_mut=config['env']['LV']['treat_death_rate_mut'],
                   growth_function_flag=config['env']['LV']['growth_function_flag'],
                   normalize=config['env']['normalize'],
                   normalize_to=config['env']['normalize_to'])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step in the environment that simulates tumor growth and treatment
        :param action: 0 - no treatment, 1 - treatment
        """
        self.state[2] = action
        # grow_tumor
        reward = 0
        for t in range(0,self.treatment_time_step):
            #step time
            self.time += 1
            self.state[0] = self.grow(0,1,self.growth_function_flag)
            self.state[1] = self.grow(1,0,self.growth_function_flag)
            self.burden = np.sum(self.state[0:2])

            # record trajectory
            self.trajectory[:,self.time-1] = self.state
            # check if done
            if self.state[0] <= 0 and self.state[1] <= 0:
                self.state = [0, 0, 0]

            if self.time >= self.max_time or self.burden >= self.threshold_burden or self.burden <= 0:
                done = True
                break
            else:
                done = False

            # get the reward
            rewards = Reward(self.reward_shaping_flag, normalization=self.threshold_burden)
            reward += rewards.get_reward(self.state, self.time/self.max_time)

        info = {}

        return [np.sum(self.state[0:2])], reward, done, info

    def render(self):
        pass

    def reset(self):
        self.real_step_count += 1

        #self.state = [self.initial_wt/self.threshold_burden, self.initial_mut/self.threshold_burden, self.initial_drug]
        if self.wt_random:
            self.initial_wt = \
            np.random.random_integers(low=0, high=self.threshold_burden_in_number, size=1)[0]
            self.initial_wt = self.initial_wt*self.threshold_burden/self.threshold_burden_in_number
        if self.mut_random:
            self.initial_mut = \
            np.random.random_integers(low=0, high=0.01*self.threshold_burden_in_number, size=1)[0]
            self.initial_mut = self.initial_mut*self.threshold_burden/self.threshold_burden_in_number

        self.state = [self.initial_wt, self.initial_mut, self.initial_drug]
        self.time = 0

        self.trajectory = np.zeros((np.shape(self.state)[0],int(self.max_time)))
        self.current_death_rate = [self.death_rate[0],self.death_rate[1]]

        return [np.sum(self.state[0:2])]

    def grow(self, i: int, j: int , flag: int) -> float:

        # adapted death model, with delay in death rate
        if flag == 0:
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
        # instantaneous death rate increase by drug application
        elif flag == 1:
            new_pop_size = self.state[i] * \
                           (1 + self.growth_rate[i] *
                            (1 - (self.state[i] + self.state[j] * self.competition[j]) / self.capacity) -
                            self.death_rate[i] -
                            self.death_rate_treat[i] * self.state[2])
        # treatment lasts certain number of tiem steps
        elif flag == 2:
            treat = self.state[2]
            if self.state[2] == 0:
                if self.time>3 and (self.trajectory[2,self.time-2]==1
                                    or self.trajectory[2,self.time-3]==1
                                    or self.trajectory[2,self.time-4]==1):
                    treat = 1
                else:
                    treat = 0
            elif self.state[2] == 1:
                if self.time>3 and (self.trajectory[2,self.time-2]==0
                                    or self.trajectory[2,self.time-3]==0):
                    treat = 0
                else:
                    treat = 1

            new_pop_size = self.state[i] * \
                           (1 + self.growth_rate[i] *
                            (1 - (self.state[i] + self.state[j] * self.competition[j]) / self.capacity) -
                            self.death_rate[i]) - self.death_rate_treat[i] * treat * self.threshold_burden

            if new_pop_size < 0:
                new_pop_size = 0

        return new_pop_size
