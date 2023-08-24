# Base environment class for all environments
import yaml
from gym import Env
from typing import Optional
from gym.spaces import Box
import numpy as np


class BaseEnv(Env):
    def __init__(
        self,
        observation_type: str = 'number',
        image_size: int = 32,
        normalize_to: int = 1,

    ) -> None:

        # Spaces
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

        self.normalize_to = normalize_to
        self.action_type = None
        self.action_space = None


        # Simulation
        self.name = 'BaseEnv'
        self.normalize = False
        self.normalize_to = None
        self.trajectory = None
        self.state = None
        self.max_tumor_size = None
        self.reward_shaping_flag = None

        # Runnning
        self.time = 0
        self.done = False


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