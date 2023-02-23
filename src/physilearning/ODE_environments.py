# imports
import yaml
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from physilearning.reward import Reward

# create environment
class LV_env(Env):
    def __init__(self, burden=1000, max_time=3000, carrying_capacity=1500,
            initial_wt=45, initial_mut=5, growth_rate_wt=0.0175, growth_rate_mut=0.0175,
            death_rate_wt=0.001, death_rate_mut=0.001, treat_death_rate_wt=0.15,
            treat_death_rate_mut=0.0, competition_wt=2.4e3, competition_mut=1.0,
            treatment_time_step=60, reward_shaping_flag=0, growth_function_flag=0):
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

        self.capacity = carrying_capacity/self.threshold_burden
        # 1 - wt, 2 - resistant
        self.growth_rate = [growth_rate_wt,growth_rate_mut]
        self.death_rate = [death_rate_wt,death_rate_mut]
        self.current_death_rate = [death_rate_wt,death_rate_mut]
        self.death_rate_treat = [treat_death_rate_wt,treat_death_rate_mut]
        self.competition = [competition_wt,competition_mut]
        self.growth_function_flag = growth_function_flag

        self.trajectory = np.zeros((np.shape(self.state)[0],int(self.max_time/self.treatment_time_step)))
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
                   growth_function_flag=growth_function_flag)
    def step(self, action):
        self.time += self.treatment_time_step
        # grow_tumor
        self.state[0] = self.grow(0,1,self.growth_function_flag)
        self.state[1] = self.grow(1,0,self.growth_function_flag)

        self.burden = np.sum(self.state[0:2])
        # do action (apply treatment or not)
        self.state[2] = action

        # record trajectory
        self.trajectory[:,int(self.time/self.treatment_time_step)-1] = self.state
        # get the reward
        rewards = Reward(self.reward_shaping_flag)
        reward = rewards.get_reward(self.state)

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

        self.trajectory = np.zeros((np.shape(self.state)[0],int(self.max_time/self.treatment_time_step)))
        self.current_death_rate = [self.death_rate[0],self.death_rate[1]]

        return self.state

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

        return new_pop_size