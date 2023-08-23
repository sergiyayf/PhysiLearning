# Base environment class for all environments
import yaml
from gym import Env
from typing import Optional

class BaseEnv(Env):
    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:

        # Configuration
        if config is None:
            self.config = self.default_config()
            self.configure(self.config)
        else:
            self.configure(config)

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None

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