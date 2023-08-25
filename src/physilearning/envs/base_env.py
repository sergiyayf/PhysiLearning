# Base environment class for all environments
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import yaml
from gym import Env
from typing import Optional
import yaml
from gym.spaces import Discrete, Box
import numpy as np
from physilearning.reward import Reward

class BaseEnv(Env):
    def __init__(
        self,
        name = 'BaseEnv',
        observation_type: str = 'number',
        action_type: str = 'discrete',
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
        env_specific_params: dict = {},
    ) -> None:
        # Normalization
        self.normalize = normalize
        self.max_tumor_size = max_tumor_size

        if self.normalize:
            self.normalization_factor = normalize_to / max_tumor_size
            self.threshold_burden = normalize_to
            self.initial_wt = initial_wt * self.normalization_factor
            self.initial_mut = initial_mut * self.normalization_factor

        else:
            self.threshold_burden = max_tumor_size
            self.initial_wt = initial_wt
            self.initial_mut = initial_mut
            self.normalization_factor = 1
        # Spaces
        self.name = name
        self.action_type = action_type
        if self.action_type == 'discrete':
            self.action_space = Discrete(2)
        elif self.action_type == 'continuous':
            self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_type = observation_type
        if self.observation_type == 'number':
            self.observation_space = Box(low=0, high=self.threshold_burden, shape=(3,))
        elif self.observation_type == 'image':
            self.observation_space = Box(low=0, high=255,
                                         shape=(1, image_size, image_size),
                                         dtype=np.uint8)
        elif self.observation_type == 'multiobs':
            raise NotImplementedError
        else:
            raise NotImplementedError
        # Image configurations
        self.image_size = image_size
        self.image = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)
        self.wt_color = 128
        self.mut_color = 255
        self.drug_color = 0
        #  Time Parameters
        self.time = 0
        self.treatment_time_step = treatment_time_step
        self.max_time = max_time

        self.initial_drug = 0
        self.burden = self.initial_mut + self.initial_wt
        self.state = [self.initial_wt,
                      self.initial_mut,
                      self.initial_drug]

        # 1 - wt, 2 - resistant
        self.growth_rate = [growth_rate_wt, growth_rate_mut]
        self.death_rate = [death_rate_wt, death_rate_mut]
        self.death_rate_treat = [treat_death_rate_wt, treat_death_rate_mut]

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

        # Other

        self.done = False
        self.reward_shaping_flag = reward_shaping_flag
        self.fig, self.ax = plt.subplots()

    @classmethod
    def from_yaml(cls, yaml_file: str, **kwargs):
        with open(yaml_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        env_name = config['env']['type']
        return cls(observation_type=config['env']['observation_type'],
                   action_type=config['env']['action_type'],
                   max_tumor_size=config['env']['max_tumor_size'],
                   max_time=config['env']['max_time'],
                   initial_wt=config['env']['initial_wt'],
                   initial_mut=config['env']['initial_mut'],
                   growth_rate_wt=config['env']['growth_rate_wt'],
                   growth_rate_mut=config['env']['growth_rate_mut'],
                   death_rate_wt=config['env']['death_rate_wt'],
                   death_rate_mut=config['env']['death_rate_mut'],
                   treat_death_rate_wt=config['env']['treat_death_rate_wt'],
                   treat_death_rate_mut=config['env']['treat_death_rate_mut'],
                   treatment_time_step=config['env']['treatment_time_step'],
                   reward_shaping_flag=config['env']['reward_shaping'],
                   normalize=config['env']['normalize'],
                   normalize_to=config['env']['normalize_to'],
                   image_size=config['env']['image_size'],
                   env_specific_params=config['env'][env_name],
                   **kwargs,
                   )

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

    def render(self, mode='human') -> mpl.animation.ArtistAnimation:
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

    def close(self):
        pass

    def seed(self, seed=None):
        raise NotImplementedError


if __name__ == '__main__':
    env = BaseEnv()