# Base environment class for all environments
import os

import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from gymnasium import Env
from gymnasium import spaces
import yaml
from gymnasium.spaces import Discrete, Box
import numpy as np
from physilearning.reward import Reward

def trunc_norm(mean, std, n_std=3):
    var = np.random.normal(mean, std)
    # truncate at n_std
    while var < mean - n_std*std or var > mean + n_std*std:
        var = np.random.normal(mean, std)
    return var

class BaseEnv(Env):
    """
    Base environment class for all environments

    :param name: Name of the environment
    :param observation_type: Type of observation space. Can be 'number', 'image', or 'multiobs'
    :param action_type: Type of action space. Can be 'discrete' or 'continuous'
    :param max_tumor_size: Maximum tumor size
    :param max_time: Maximum time for the environment
    :param initial_wt: Initial wild-type tumor size
    :param initial_mut: Initial mutant tumor size
    :param growth_rate_wt: Growth rate of wild-type tumor
    :param growth_rate_mut: Growth rate of mutant tumor
    :param death_rate_wt: Death rate of wild-type tumor
    :param death_rate_mut: Death rate of mutant tumor
    :param treat_death_rate_wt: Death rate of wild-type tumor under treatment
    :param treat_death_rate_mut: Death rate of mutant tumor under treatment
    :param treatment_time_step: Time step for treatment
    :param reward_shaping_flag: Flag for reward shaping.
    :param normalize: Flag for normalization. Can be 0 or 1
    :param normalize_to: Maximum tumor size to normalize to
    :param image_size: Size of the image
    :param kwargs: Additional arguments

    """
    def __init__(
        self,
        config: dict = None,
        name: str = 'BaseEnv',
        observation_type: str = 'number',
        action_type: str = 'discrete',
        see_resistance: bool = False,
        see_prev_action: bool = False,
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
        reward_shaping_flag: str = 'ttp',
        normalize: bool = 1,
        normalize_to: float = 1000,
        image_size: int = 84,
        patient_id: int | list = 0,
        **kwargs,
    ) -> None:
        self.config = config
        if isinstance(patient_id, list):
            self.patient_id_list = patient_id
            if self.config['env']['patient_sampling']['type'] == 'range':
                self.patient_id = 1
            else:
                self.patient_id = np.random.choice(patient_id)
        elif isinstance(patient_id, int):
            self.patient_id_list = [patient_id]
            self.patient_id = patient_id
        else:
            raise ValueError("patient_id must be an integer or a list of integers")
        # Normalization
        self.normalize = normalize
        self.normalize_to = normalize_to
        self.max_tumor_size = max_tumor_size
        self.initial_wt = initial_wt
        self.initial_mut = initial_mut
        self.growth_rate = [growth_rate_wt, growth_rate_mut]
        self.death_rate = [death_rate_wt, death_rate_mut]
        self.death_rate_treat = [treat_death_rate_wt, treat_death_rate_mut]
        if self.config is not None:
            self.capacity = self.config['env']['LvEnv']['carrying_capacity']
        else:
            self.capacity = 1
        self.random_params = self.set_random_params()
        self.randomize_params()
        if self.normalize:
            self.normalization_factor = normalize_to / (self.initial_mut + self.initial_wt)
            self.threshold_burden = normalize_to*max_tumor_size
            self.initial_wt *= self.normalization_factor
            self.initial_mut *= self.normalization_factor

        else:
            self.threshold_burden = max_tumor_size*(self.initial_mut + self.initial_wt)
            self.normalization_factor = 1

        # Spaces
        self.name = name
        self.action_type = action_type
        self.see_resistance = see_resistance
        self.see_prev_action = see_prev_action
        if self.action_type == 'discrete':
            self.action_space = Discrete(2)
        elif self.action_type == 'continuous':
            self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_type = observation_type
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

        self.state = [self.initial_wt,
                      self.initial_mut,
                      self.initial_drug]

        # 1 - wt, 2 - resistant

        # If patient sampling enabled set patient specific parameters
        if self.config is not None:
            if self.config['env']['patient_sampling']['enable']:
                self._set_patient_params()
        else:
            self.config = {'env': {'patient_sampling': {'enable': False}}}

        if self.observation_type == 'number':
            num_obs = 1
            if self.see_prev_action:
                num_obs+=1
            if see_resistance:
                num_obs+=1
            self.observation_space = Box(low=0, high=self.threshold_burden, shape=(num_obs,))

        else:
            self.image_trajectory = np.zeros(
                (self.image_size, self.image_size, int(self.max_time / self.treatment_time_step) + 1))
            self.image_trajectory[:, :, 0] = self.image[0, :, :]
            if self.observation_type == 'image':
                self.observation_space = Box(low=0, high=255,
                                             shape=(1, image_size, image_size),
                                             dtype=np.uint8)
            elif self.observation_type == 'multiobs':
                self.observation_space = spaces.Dict(
                    spaces={
                        "vec": spaces.Box(low=0, high=self.threshold_burden, shape=(3,)),
                        "img": spaces.Box(low=0, high=255,
                                          shape=(1, image_size, image_size),
                                          dtype=np.uint8)
                            }
                )
            elif self.observation_type == 'mutant_position':
                if see_resistance:
                    self.observation_space = Box(low=0, high=self.threshold_burden, shape=(4,))
                else:
                    self.observation_space = Box(low=0, high=self.threshold_burden, shape=(3,))
            else:
                raise NotImplementedError

        if self.observation_type == 'mutant_position':
            self.mutant_normalized_position = 0
            self.trajectory = np.zeros((np.shape(self.state)[0]+2, int(self.max_time) + 1))
            self.trajectory[0:3, 0] = self.state
            self.trajectory[3, 0] = self.mutant_normalized_position
            self.trajectory[4, 0] = 0
        else:
            self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time) + 1))
            self.trajectory[:, 0] = self.state

        # Other
        self.done = False
        self.reward_shaping_flag = reward_shaping_flag
        # self.fig, self.ax = plt.subplots()

    @classmethod
    def from_yaml(cls, yaml_file: str, **kwargs):
        with open(yaml_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        env_name = config['env']['type']
        return cls(config=config,
                   observation_type=config['env']['observation_type'],
                   action_type=config['env']['action_type'],
                   see_resistance=config['env']['see_resistance'],
                   see_prev_action=config['env']['see_prev_action'],
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
                   patient_id=config['env']['patient_sampling']['patient_id'],
                   env_specific_params=config['env'][env_name],
                   **kwargs,
                   )

    def _set_patient_params(self):
        """
        Set parameters of presimulated patients
        """
        self.initial_mut = self.config['patients'][self.patient_id]['initial_mut']
        self.initial_wt = self.config['patients'][self.patient_id]['initial_wt']
        if self.normalize:
            self.normalization_factor = self.normalize_to / (self.initial_mut + self.initial_wt)
            self.initial_wt = self.initial_wt * self.normalization_factor
            self.initial_mut = self.initial_mut * self.normalization_factor

        self.growth_rate = [self.config['patients'][self.patient_id]['growth_rate_wt'],
                            self.config['patients'][self.patient_id]['growth_rate_mut']]
        self.death_rate = [self.config['patients'][self.patient_id]['death_rate_wt'],
                           self.config['patients'][self.patient_id]['death_rate_mut']]
        self.death_rate_treat = [self.config['patients'][self.patient_id]['treat_death_rate_wt'],
                                 self.config['patients'][self.patient_id]['treat_death_rate_mut']]

        return

    def set_random_params(self):
        random_params = {}
        if isinstance(self.initial_wt, str):
            random_params['initial_wt'] = self.initial_wt
        if isinstance(self.initial_mut, str):
            random_params['initial_mut'] = self.initial_mut
        if isinstance(self.growth_rate[0], str):
            random_params['growth_rate_wt'] = self.growth_rate[0]
        if isinstance(self.growth_rate[1], str):
            random_params['growth_rate_mut'] = self.growth_rate[1]
        if isinstance(self.death_rate_treat[0], str):
            random_params['death_rate_treat_wt'] = self.death_rate_treat[0]
        if isinstance(self.death_rate_treat[1], str):
            random_params['death_rate_treat_mut'] = self.death_rate_treat[1]
        if isinstance(self.capacity, str):
            random_params['capacity'] = self.capacity
        return random_params

    def randomize_params(self):
        for key, value in self.random_params.items():
            if '-' in value:
                low, high = (int(val) for val in value.split('-'))
                val = np.random.randint(low, high)
            elif 'pm' in value:
                mean = int(value.split('pm')[0])
                std = int(value.split('pm')[1])
                val = trunc_norm(mean, std, 3)
            else:
                val = value
            if key == 'initial_wt':
                self.initial_wt = val
            if key == 'initial_mut':
                self.initial_mut = val
            if key == 'growth_rate_wt':
                self.growth_rate[0] = val
            if key == 'growth_rate_mut':
                if value == 'same':
                    self.growth_rate[1] = self.growth_rate[0]
                else:
                    self.growth_rate[1] = val
            if key == 'death_rate_treat_wt':
                self.death_rate_treat[0] = val
            if key == 'capacity':
                self.capacity = val

    def _choose_new_patient(self):
        """
        Choose new patient from the list of patients
        """
        if self.config['env']['patient_sampling']['type'] == 'random':
            self.patient_id = np.random.choice(self.config['env']['patient_sampling']['patient_id'])
            self._set_patient_params()
        elif self.config['env']['patient_sampling']['type'] == 'sequential':
            patient_list = self.config['env']['patient_sampling']['patient_id']
            patient_index = patient_list.index(self.patient_id)
            if patient_index == len(patient_list) - 1:
                self.patient_id = patient_list[0]
            else:
                self.patient_id = patient_list[patient_index + 1]
            self._set_patient_params()
        elif self.config['env']['patient_sampling']['type'] == 'range':
            range_values = self.config['env']['patient_sampling']['patient_id']
            patient_list = list(range(range_values[0], range_values[1]))
            patient_index = patient_list.index(self.patient_id)
            if patient_index == len(patient_list) - 1:
                self.patient_id = patient_list[0]
            else:
                self.patient_id = patient_list[patient_index + 1]

        else:
            raise ValueError('Patient sampling type not supported')

    def measure_response(self):
        """
        Measure response to treatment
        """
        if self.trajectory[2, self.time] == 1:
            response = np.sum(self.trajectory[0:2, self.time-1])-np.sum(self.state[0:2])
        else:
            response = 0
        return response

    def truncate(self):
        """
        Truncation condition
        """
        if self.time >= self.max_time:
            truncate = True
        else:
            truncate = False
        # truncate = False
        return truncate

    def terminate(self):
        """
        Termination condition
        """
        if np.sum(self.state[0:2]) >= self.threshold_burden:
            terminate = True
        else:
            terminate = False
        # terminate = False
        return terminate

    @classmethod
    def default_config(cls) -> dict: # pragma: no cover
        """
        Default configuration for environment
        Returns
        -------

        """
        pass

    def configure(self, config: dict) -> None: # pragma: no cover
        """
        Configure environment
        Parameters
        ----------
        config

        Returns
        -------

        """
        pass

    def step(self, action): # pragma: no cover
        raise NotImplementedError

    def reset(self, *, seed=None, options=None): # pragma: no cover
        raise NotImplementedError

    def get_reward(self):
        rewards = Reward(self.reward_shaping_flag, normalization=np.sum(self.trajectory[0:2, 0]))

        if self.reward_shaping_flag == 'tendayaverage':
            reward = rewards.tendayaverage(self.trajectory, self.time)
        elif self.reward_shaping_flag == 'mtd_compare':
            reward = rewards.tendayaverage(self.trajectory, self.time)
        else:
            reward = rewards.get_reward(self.state, self.time, self.trajectory)
        return reward
    def render(self, mode='human') -> mpl.animation.ArtistAnimation:
        """
        Render environment
        Produce animation of tumor growth if observation_type is 'image' or 'multiobs'
        Parameters
        ----------
        mode

        Returns
        -------

        """
        self.fig, self.ax = plt.subplots()
        if self.observation_type == 'number':
            pass
        elif self.observation_type == 'image' or self.observation_type == 'multiobs':
            ims = []

            for i in range(self.time):
                im = self.ax.imshow(self.image_trajectory[:, :, i], animated=True, cmap='viridis', vmin=0, vmax=255)
                ims.append([im])
            ani = animation.ArtistAnimation(self.fig, ims, interval=2.1, blit=True, repeat_delay=1000)

            return ani

    def close(self): # pragma: no cover
        pass

    def seed(self, seed=None): # pragma: no cover
        raise NotImplementedError


if __name__ == '__main__': # pragma: no cover
    os.chdir('/home/saif/Projects/PhysiLearning')
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['env']['patient_sampling']['enable'] = False
    env = BaseEnv(config=config)
    print(env.initial_wt)