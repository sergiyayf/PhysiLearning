import yaml
from gym.spaces import Discrete, Box
import numpy as np
from physilearning.envs.base_env import BaseEnv
from physilearning.reward import Reward
from typing import Tuple
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt


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
        observation_type: str = 'number',
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
        normalize_to: float = 1000,
        image_size: int = 84,
        image_sampling_type: str = 'random',
    ) -> None:
        # Spaces
        self.name = 'LvEnv'
        self.action_space = Discrete(2)
        self.observation_type = observation_type
        if self.observation_type == 'number':
            self.observation_space = Box(low=0,high=normalize_to,shape=(3,))
        elif self.observation_type == 'image':
            self.observation_space = Box(low=0, high=255,
                                         shape=(1, image_size, image_size),
                                         dtype=np.uint8)
        elif self.observation_type == 'multiobs':
            raise NotImplementedError
        else:
            raise NotImplementedError

        # Configurations
        self.image_size = image_size
        self.image = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)

        #  Time Parameters
        self.time = 0
        self.treatment_time_step = treatment_time_step
        self.max_time = max_time

        # Normalization
        self.normalize = normalize
        self.max_tumor_size = max_tumor_size

        # Check if initial_wt and initial_mut are random
        self.wt_random = isinstance(initial_wt, str)
        if self.wt_random:
            self.initial_wt = np.random.random_integers(low=0, high=int(0.99*self.max_tumor_size), size=1)[0]
        else:
            self.initial_wt = initial_wt
        self.mut_random = isinstance(initial_mut, str)
        if self.mut_random:
            self.initial_mut = np.random.random_integers(low=0, high=int(0.01*self.max_tumor_size), size=1)[0]
        else:
            self.initial_mut = initial_mut

        # Normalizazion
        if self.normalize:
            self.normalization_factor = normalize_to / max_tumor_size
            self.initial_wt = self.initial_wt*self.normalization_factor
            self.initial_mut = self.initial_mut*self.normalization_factor
            self.threshold_burden = normalize_to
            self.capacity = carrying_capacity*self.normalization_factor
        else:
            self.normalization_factor = 1
            self.threshold_burden = max_tumor_size
            self.capacity = carrying_capacity


        self.initial_drug = 0
        self.burden = self.initial_mut+self.initial_wt
        self.state = [self.initial_wt,
                      self.initial_mut,
                      self.initial_drug]

        # 1 - wt, 2 - resistant
        self.growth_rate = [growth_rate_wt,growth_rate_mut]
        self.death_rate = [death_rate_wt,death_rate_mut]
        self.current_death_rate = [death_rate_wt,death_rate_mut]
        self.death_rate_treat = [treat_death_rate_wt,treat_death_rate_mut]
        self.competition = [competition_wt,competition_mut]
        self.growth_function_flag = growth_function_flag

        # trajectory for plotting
        if self.observation_type == 'number':
            self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time / self.treatment_time_step) + 1))
        elif self.observation_type == 'image':
            self.image_trajectory = np.zeros(
                (self.image_size, self.image_size, int(self.max_time / self.treatment_time_step) + 1))
            self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time / self.treatment_time_step) + 1))
            self.image_trajectory[:, :, 0] = self.image[0, :, :]

        self.trajectory = np.zeros((np.shape(self.state)[0],int(self.max_time)+1))
        self.trajectory[:,0] = self.state
        self.real_step_count = 0
        self.wt_color = 128
        self.mut_color = 255
        self.drug_color = 0
        self.done = False

        self.reward_shaping_flag = reward_shaping_flag
        self.fig, self.ax = plt.subplots()
        self.image_sampling_type = image_sampling_type

    @classmethod
    def from_yaml(cls, yaml_file: str):
        with open(yaml_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return cls(max_tumor_size=config['env']['max_tumor_size'],
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
                   normalize_to=config['env']['normalize_to'],
                   image_size=config['env']['image_size'],
                   observation_type=config['env']['observation_type'],
                   image_sampling_type=config['env']['LV']['image_sampling_type'],
                   )


    def _get_image(self, action: int):
        """
        Randomly sample a tumor inside of the image and return the image
        """
        # estimate the number of cells to sample
        num_wt_to_sample = self.image_size * self.image_size * \
                           self.state[0] / (self.capacity * self.normalization_factor)
        num_mut_to_sample = self.image_size * self.image_size * \
                            self.state[1] / (self.capacity * self.normalization_factor)

        if self.image_sampling_type == 'random':

            # Sample sensitive clones
            random_indices = np.random.choice(self.image_size*self.image_size,
                                              int(np.round(num_wt_to_sample)), replace=False)
            wt_x, wt_y = np.unravel_index(random_indices, (self.image_size, self.image_size))

            # Sample resistant clones
            random_indices = np.random.choice(self.image_size*self.image_size,
                                                int(np.round(num_mut_to_sample)), replace=False)
            mut_x, mut_y = np.unravel_index(random_indices, (self.image_size, self.image_size))

        elif self.image_sampling_type == 'dense':

            wt_x, wt_y = [], []
            mut_x, mut_y = [], []
            radius = int(np.sqrt(num_wt_to_sample+num_mut_to_sample)/3)

            while len(wt_x) < num_wt_to_sample:
                x = np.random.randint(0,self.image_size)
                y = np.random.randint(0,self.image_size)
                if np.sqrt((x-self.image_size/2)**2 + (y-self.image_size/2)**2) < radius:
                    wt_x.append(x)
                    wt_y.append(y)

            while len(mut_x) < num_mut_to_sample:
                x = np.random.randint(0,self.image_size)
                y = np.random.randint(0,self.image_size)
                if np.sqrt((x-self.image_size/2)**2 + (y-self.image_size/2)**2) < radius:
                    mut_x.append(x)
                    mut_y.append(y)

        # populate the image
        # clean the image and make the new one
        if action:
            self.image = self.drug_color * np.ones((1, self.image_size, self.image_size), dtype=np.uint8)
        else:
            self.image = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)

        for x, y in zip(wt_x, wt_y):
            self.image[0, int(x), int(y)] = self.wt_color
        for x, y in zip(mut_x, mut_y):
            self.image[0, int(x), int(y)] = self.mut_color

        return self.image

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step in the environment that simulates tumor growth and treatment
        :param action: 0 - no treatment, 1 - treatment
        """

        # grow_tumor
        reward = 0
        for t in range(0,self.treatment_time_step):
            #step time
            self.time += 1
            self.state[0] = self.grow(0,1,self.growth_function_flag)
            self.state[1] = self.grow(1,0,self.growth_function_flag)
            self.burden = np.sum(self.state[0:2])

            # record trajectory
            self.state[2] = action
            self.trajectory[:,self.time] = self.state

            # check if done
            if self.state[0] <= 0 and self.state[1] <= 0:
                self.state = [0, 0, 0]

            if self.time >= self.max_time-1 or self.burden >= self.threshold_burden or self.burden <= 0:
                done = True
                break
            else:
                done = False

            # get the reward
            rewards = Reward(self.reward_shaping_flag, normalization=self.threshold_burden)
            reward += rewards.get_reward(self.state, self.time/self.max_time)

        info = {}

        if self.observation_type == 'number':
            obs = self.state
        elif self.observation_type == 'image':
            self.image = self._get_image(action)
            self.image_trajectory[:, :, int(self.time/self.treatment_time_step)] = self.image[0, :, :]
            obs = self.image
        else:
            obs = None
            raise NotImplementedError
        self.done = done

        return obs, reward, done, info


    def reset(self):
        self.real_step_count += 1

        #self.state = [self.initial_wt/self.threshold_burden, self.initial_mut/self.threshold_burden, self.initial_drug]
        if self.wt_random:
            self.initial_wt = \
            np.random.random_integers(low=0, high=self.max_tumor_size, size=1)[0]
            if self.normalize:
                self.initial_wt = self.initial_wt*self.normalization_factor
        if self.mut_random:
            self.initial_mut = \
            np.random.random_integers(low=0, high=0.01*self.max_tumor_size, size=1)[0]
            if self.normalize:
                self.initial_mut = self.initial_mut*self.normalization_factor

        self.state = [self.initial_wt, self.initial_mut, self.initial_drug]
        self.time = 0

        self.trajectory = np.zeros((np.shape(self.state)[0],int(self.max_time)+1))
        self.trajectory[:,0] = self.state

        if self.observation_type == 'number':
            obs = self.state
        elif self.observation_type == 'image':
            self.image = self._get_image(self.initial_drug)
            self.image_trajectory = np.zeros(
                (self.image_size, self.image_size, int(self.max_time / self.treatment_time_step) + 1))
            self.image_trajectory[:, :, 0] = self.image[0, :, :]
            obs = self.image
        else:
            obs = None
            raise NotImplementedError

        self.current_death_rate = [self.death_rate[0],self.death_rate[1]]

        return obs

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
        # one time step delay in treatment effect
        elif flag == 2:
            treat = self.state[2]
            if self.state[2] == 0:
                if self.time>1 and (self.trajectory[2,self.time-1]==1):
                    treat = 1
                else:
                    treat = 0
            elif self.state[2] == 1:
                if self.time>1 and (self.trajectory[2,self.time-1]==0):
                    treat = 0
                else:
                    treat = 1
            new_pop_size = self.state[i] * \
                           (1 + self.growth_rate[i] *
                            (1 - (self.state[i] + self.state[j] * self.competition[j]) / self.capacity) -
                            self.death_rate[i] - self.death_rate_treat[i] * treat)

            if new_pop_size < 0:
                new_pop_size = 0


        return new_pop_size


    def render(self, mode: str = 'human') -> mpl.animation.ArtistAnimation:
        # render state
        # plot it on the grid with different colors for wt and mut
        # animate simulation with matplotlib animation

        if self.observation_type == 'number':
            pass
        elif self.observation_type == 'image':
            ims = []

            for i in range(self.time):
                im = self.ax.imshow(self.image_trajectory[:, :, i], animated=True, cmap='viridis', vmin=0, vmax=255)
                ims.append([im])
            ani = animation.ArtistAnimation(self.fig, ims, interval=2.1, blit=True, repeat_delay=1000)

            return ani


if __name__ == "__main__":
    env = LvEnv.from_yaml("../../../config.yaml")
    env.reset()
    grid = env.image

    while not env.done:
        act = 1  # env.action_space.sample()
        env.step(act)

    anim = env.render()
