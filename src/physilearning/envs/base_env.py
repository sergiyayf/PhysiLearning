# Base environment class for all environments
import yaml
from gym import Env
from typing import Optional
from gym.spaces import Box
import numpy as np
import yaml
from gym.spaces import Discrete, Box
import numpy as np
from physilearning.envs.base_env import BaseEnv
from physilearning.reward import Reward
from typing import Tuple
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt

class BaseEnv(Env):
    def __init__(
        self,
        observation_type: str = 'number',
        max_tumor_size: float = 1000,
        max_time: int = 3000,
        initial_wt: float = 45,
        initial_mut: float = 5,
        growth_rate_wt: float = 0.0175,
        growth_rate_mut: float = 0.0175,
        death_rate_wt: float = 0.001,
        death_rate_mut: float = 0.001,
        treat_death_rate_wt: float = 0.15,
        treat_death_rate_mut: float = 0.0,
        treatment_time_step: int = 60,
        reward_shaping_flag: int = 0,
        normalize: bool = 1,
        normalize_to: float = 1000,
        image_size: int = 84,
        image_sampling_type: str = 'random',
    ) -> None:
        super().__init__()

        # Spaces
        self.name = 'LvEnv'
        self.action_space = Discrete(2)
        self.observation_type = observation_type
        if self.observation_type == 'number':
            self.observation_space = Box(low=0, high=normalize_to, shape=(3,))
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
            self.initial_wt = np.random.random_integers(low=0, high=int(0.99 * self.max_tumor_size), size=1)[0]
        else:
            self.initial_wt = initial_wt
        self.mut_random = isinstance(initial_mut, str)
        if self.mut_random:
            self.initial_mut = np.random.random_integers(low=0, high=int(0.01 * self.max_tumor_size), size=1)[0]
        else:
            self.initial_mut = initial_mut

        # Normalizazion
        if self.normalize:
            self.normalization_factor = normalize_to / max_tumor_size
            self.initial_wt = self.initial_wt * self.normalization_factor
            self.initial_mut = self.initial_mut * self.normalization_factor
            self.threshold_burden = normalize_to
            self.capacity = carrying_capacity * self.normalization_factor
        else:
            self.normalization_factor = 1
            self.threshold_burden = max_tumor_size
            self.capacity = carrying_capacity

        self.initial_drug = 0
        self.burden = self.initial_mut + self.initial_wt
        self.state = [self.initial_wt,
                      self.initial_mut,
                      self.initial_drug]

        # 1 - wt, 2 - resistant
        self.growth_rate = [growth_rate_wt, growth_rate_mut]
        self.death_rate = [death_rate_wt, death_rate_mut]
        self.current_death_rate = [death_rate_wt, death_rate_mut]
        self.death_rate_treat = [treat_death_rate_wt, treat_death_rate_mut]
        self.competition = [competition_wt, competition_mut]
        self.growth_function_flag = growth_function_flag

        # trajectory for plotting
        if self.observation_type == 'number':
            self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time / self.treatment_time_step) + 1))
        elif self.observation_type == 'image':
            self.image_trajectory = np.zeros(
                (self.image_size, self.image_size, int(self.max_time / self.treatment_time_step) + 1))
            self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time / self.treatment_time_step) + 1))
            self.image_trajectory[:, :, 0] = self.image[0, :, :]

        self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time) + 1))
        self.trajectory[:, 0] = self.state
        self.real_step_count = 0
        self.wt_color = 128
        self.mut_color = 255
        self.drug_color = 0
        self.done = False

        self.reward_shaping_flag = reward_shaping_flag
        self.fig, self.ax = plt.subplots()
        self.image_sampling_type = image_sampling_type

    @classmethod
    def from_yaml(cls, yaml_file, kwargs=None):
        """
        Load environment from yaml file
        Parameters
        ----------
        yaml_file

        Returns
        -------
        Environment object from yaml file
        """
        with open(yaml_file, 'r') as f:
            config  = yaml.load(f, Loader=yaml.FullLoader)

        return cls(config)

    @classmethod
    def default_config(cls) -> dict:
        """
        Default configuration for environment
        Returns
        -------

        """
        pass

    def configure(self, config: dict) -> None:
        """
        Configure environment
        Parameters
        ----------
        config

        Returns
        -------

        """
        pass
    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self, seed=None):
        raise NotImplementedError