# imports
import yaml
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from physilearning.envs.base_env import BaseEnv
from physilearning.reward import Reward

# create environment
class LvEnv(BaseEnv):
    def __init__(self, burden=1000, max_time=3000, carrying_capacity=1500,
            initial_wt=45, initial_mut=5, growth_rate_wt=0.0175, growth_rate_mut=0.0175,
            death_rate_wt=0.001, death_rate_mut=0.001, treat_death_rate_wt=0.15,
            treat_death_rate_mut=0.0, competition_wt=2.4e3, competition_mut=1.0,
            treatment_time_step=60, reward_shaping_flag=0, growth_function_flag=0,normalize_to=1000):
        # setting up environment
        # set up discrete action space
        self.name = 'LvEnv'
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0,high=normalize_to,shape=(1,))
        self.time = 0
        self.treatment_time_step = treatment_time_step
        self.max_time = max_time
        self.threshold_burden_in_number = burden
        self.threshold_burden = normalize_to
        self.wt_random = isinstance(initial_wt, str)
        if self.wt_random:
            self.initial_wt = np.random.random_integers(low=0, high=self.threshold_burden_in_number, size=1)[0]
            self.initial_wt = self.initial_wt*self.threshold_burden/self.threshold_burden_in_number
        else:
            self.initial_wt = initial_wt*normalize_to/burden
        self.mut_random = isinstance(initial_mut, str)
        if self.mut_random:
            self.initial_mut = np.random.random_integers(low=0, high=0.01*self.threshold_burden_in_number, size=1)[0]
            self.initial_mut = self.initial_mut*self.threshold_burden/self.threshold_burden_in_number
        else:
            self.initial_mut = initial_mut*normalize_to/burden
        self.initial_drug = 0
        self.burden = self.initial_mut+self.initial_wt
        # self.state = [self.initial_wt/self.threshold_burden,
        #               self.initial_mut/self.threshold_burden,
        #               self.initial_drug]
        # self.capacity = carrying_capacity / self.threshold_burden
        self.state = [self.initial_wt,
                      self.initial_mut,
                      self.initial_drug]
        self.capacity = carrying_capacity*normalize_to/burden
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
    def from_yaml(cls, yaml_file, port='0', job_name='000000'):
        with open(yaml_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # general env settings
        burden = config['env']['threshold_burden']
        max_time = config['env']['max_time']
        timestep = config['env']['treatment_time_step']
        reward_shaping_flag = config['env']['reward_shaping']
        normalize_to = config['env']['normalize_to']

        # LV specific settings
        initial_wt = config['env']['LV']['initial_wt']
        initial_mut = config['env']['LV']['initial_mut']
        carrying_capacity = config['env']['LV']['carrying_capacity']
        growth_rate_wt = config['env']['LV']['growth_rate_wt']
        growth_rate_mut = config['env']['LV']['growth_rate_mut']
        death_rate_wt = config['env']['LV']['death_rate_wt']
        death_rate_mut = config['env']['LV']['death_rate_mut']
        treat_death_rate_wt = config['env']['LV']['treat_death_rate_wt']
        treat_death_rate_mut = config['env']['LV']['treat_death_rate_mut']
        competition_wt = config['env']['LV']['competition_wt']
        competition_mut = config['env']['LV']['competition_mut']
        growth_function_flag = config['env']['LV']['growth_function_flag']

        # global settings
        transport_type = config['global']['transport_type']
        transport_address = config['global']['transport_address']
        if transport_type == 'ipc://':
            transport_address = f'{transport_address}{job_name}{port}'
        else:
            warnings.warn('Transport type is different from ipc, please check the config file if everything is correct')
            transport_address = f'{transport_address}:{port}'

        return cls(burden=burden, max_time=max_time,
                   initial_wt=initial_wt, treatment_time_step=timestep, initial_mut=initial_mut,
                   reward_shaping_flag=reward_shaping_flag, carrying_capacity=carrying_capacity,
                   growth_rate_wt=growth_rate_wt, growth_rate_mut=growth_rate_mut,
                   death_rate_wt=death_rate_wt, death_rate_mut=death_rate_mut,
                   competition_wt=competition_wt, competition_mut=competition_mut,
                   treat_death_rate_wt=treat_death_rate_wt, treat_death_rate_mut=treat_death_rate_mut,
                   growth_function_flag=growth_function_flag, normalize_to=normalize_to)
    def step(self, action):
        # do action (apply treatment or not)
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

    def grow(self, i, j, flag):  # i index of growing type, j: index of competing type

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
