import numpy as np
from physilearning.envs.base_env import BaseEnv
from physilearning.reward import Reward
from typing import Tuple


class LvEnv(BaseEnv):
    """
    Environment for Lottka-Volterra tumor growth model

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
    :param env_specific_params: Dictionary of environment specific parameters
    :param kwargs: Additional arguments
    """

    def __init__(
        self,
        config: dict = None,
        name: str = 'LvEnv',
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
        normalize:  bool = 1,
        normalize_to: float = 1000,
        image_size: int = 84,
        patient_id: int | list = 0,
        env_specific_params: dict = {},
        **kwargs,
    ) -> None:

        self.wt_random = isinstance(initial_wt, str)
        self.mut_random = isinstance(initial_mut, str)
        if self.wt_random:
            initial_wt = np.random.random_integers(low=0, high=int(0.99*max_tumor_size), size=1)[0]
        if self.mut_random:
            initial_mut = np.random.random_integers(low=0, high=int(0.99*max_tumor_size), size=1)[0]
        super().__init__(config=config, name=name, observation_type=observation_type, action_type=action_type,
                         max_tumor_size=max_tumor_size, max_time=max_time, initial_wt=initial_wt,
                         initial_mut=initial_mut, growth_rate_wt=growth_rate_wt, growth_rate_mut=growth_rate_mut,
                         death_rate_wt=death_rate_wt, death_rate_mut=death_rate_mut,
                         treat_death_rate_wt=treat_death_rate_wt, treat_death_rate_mut=treat_death_rate_mut,
                         treatment_time_step=treatment_time_step, reward_shaping_flag=reward_shaping_flag,
                         normalize=normalize, normalize_to=normalize_to, image_size=image_size, patient_id=patient_id,
                         )

        # Normalizazion
        if self.normalize:
            self.capacity = env_specific_params.get('carrying_capacity', 6500) \
                            * self.normalization_factor
        else:
            self.capacity = env_specific_params.get('carrying_capacity', 6500)

        # 1 - wt, 2 - resistant
        self.competition = [env_specific_params.get('competition_wt', 2.),
                            env_specific_params.get('competition_mut', 1.)]
        self.growth_function_flag = env_specific_params.get('growth_function_flag', 'delayed')

        self.trajectory[:, 0] = self.state
        self.real_step_count = 0

        self.image_sampling_type = env_specific_params.get('image_sampling_type', 'random')

    def _get_image(self, action: int):
        """
        Randomly sample a tumor inside of the image and return the image
        """
        # estimate the number of cells to sample
        num_wt_to_sample = np.round(self.image_size * self.image_size * \
            self.state[0] / (self.capacity * self.normalization_factor))
        num_mut_to_sample = np.round(self.image_size * self.image_size * \
            self.state[1] / (self.capacity * self.normalization_factor))

        if self.image_sampling_type == 'random':

            # Sample sensitive clones
            random_indices = np.random.choice(self.image_size*self.image_size,
                                              int(num_wt_to_sample), replace=False)
            wt_x, wt_y = np.unravel_index(random_indices, (self.image_size, self.image_size))

            # Sample resistant clones
            random_indices = np.random.choice(self.image_size*self.image_size,
                                              int(num_mut_to_sample), replace=False)
            mut_x, mut_y = np.unravel_index(random_indices, (self.image_size, self.image_size))

        elif self.image_sampling_type == 'dense':

            wt_x, wt_y = [], []
            mut_x, mut_y = [], []
            radius = np.round(np.sqrt(num_wt_to_sample+num_mut_to_sample)/2.2*np.sqrt(2)+1)

            while len(wt_x) < num_wt_to_sample:
                x = np.random.randint(0, self.image_size)
                y = np.random.randint(0, self.image_size)
                if np.sqrt((x-self.image_size/2)**2 + (y-self.image_size/2)**2) < radius:
                    # check if pair is already in the list
                    if (x,y) not in zip(wt_x, wt_y):
                        wt_x.append(x)
                        wt_y.append(y)

            while len(mut_x) < num_mut_to_sample:
                x = np.random.randint(0, self.image_size)
                y = np.random.randint(0, self.image_size)
                if np.sqrt((x-self.image_size/2)**2 + (y-self.image_size/2)**2) < radius:
                    if (x, y) not in zip(wt_x, wt_y):
                        mut_x.append(x)
                        mut_y.append(y)
        else:
            raise ValueError('Unknown image sampling type')
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
        for t in range(0, self.treatment_time_step):
            # step time
            self.time += 1
            self.state[0] = self.grow(0, 1, self.growth_function_flag)
            self.state[1] = self.grow(1, 0, self.growth_function_flag)
            self.burden = np.sum(self.state[0:2])

            # record trajectory
            self.state[2] = action
            self.trajectory[:, self.time] = self.state

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
        elif self.observation_type == 'image' or self.observation_type == 'multiobs':
            self.image = self._get_image(action)
            self.image_trajectory[:, :, int(self.time/self.treatment_time_step)] = self.image[0, :, :]
            if self.observation_type == 'image':
                obs = self.image
            elif self.observation_type == 'multiobs':
                obs = {'vec': self.state, 'img': self.image}
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self.done = done

        return obs, reward, done, info

    def reset(self):
        self.real_step_count += 1
        if self.config['env']['patient_sampling']['enable']:
            if len(self.patient_id_list) > 1:
                self._choose_new_patient()

        if self.wt_random:
            self.initial_wt = \
                np.random.random_integers(low=0, high=int(self.max_tumor_size), size=1)[0]
            if self.normalize:
                self.initial_wt = self.initial_wt*self.normalization_factor
        if self.mut_random:
            self.initial_mut = \
                np.random.random_integers(low=0, high=int(0.01*self.max_tumor_size), size=1)[0]
            if self.normalize:
                self.initial_mut = self.initial_mut*self.normalization_factor

        self.state = [self.initial_wt, self.initial_mut, self.initial_drug]
        self.time = 0

        self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time)+1))
        self.trajectory[:, 0] = self.state

        if self.observation_type == 'number':
            obs = self.state
        elif self.observation_type == 'image' or self.observation_type == 'multiobs':
            self.image = self._get_image(self.initial_drug)
            self.image_trajectory = np.zeros(
                (self.image_size, self.image_size, int(self.max_time / self.treatment_time_step) + 1))
            self.image_trajectory[:, :, 0] = self.image[0, :, :]
            if self.observation_type == 'image':
                obs = self.image
            elif self.observation_type == 'multiobs':
                obs = {'vec': self.state, 'img': self.image}
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return obs

    def grow(self, i: int, j: int, flag: str) -> float:

        # instantaneous death rate increase by drug application
        if flag == 'instant':
            new_pop_size = self.state[i] * \
                           (1 + self.growth_rate[i] *
                            (1 - (self.state[i] + self.state[j] * self.competition[j]) / self.capacity) -
                            self.death_rate[i] -
                            self.death_rate_treat[i] * self.state[2])
        # one time step delay in treatment effect
        elif flag == 'delayed':
            treat = self.state[2]
            if self.state[2] == 0:
                if self.time > 1 and (self.trajectory[2, self.time-1] == 1):
                    treat = 1
                else:
                    treat = 0
            elif self.state[2] == 1:
                if self.time > 1 and (self.trajectory[2, self.time-1] == 0):
                    treat = 0
                else:
                    treat = 1
            new_pop_size = self.state[i] * (1 + self.growth_rate[i] *
                (1 - (self.state[i] + self.state[j] * self.competition[j]) / self.capacity) -
                self.death_rate[i] - self.death_rate_treat[i] * treat)

            if new_pop_size < 0.1*self.normalization_factor:
                new_pop_size = 0
        else:
            raise NotImplementedError
        return new_pop_size


if __name__ == "__main__":
    # set random seed
    np.random.seed(0)
    env = LvEnv.from_yaml("../../../config.yaml")
    env.reset()
    grid = env.image

    while not env.done:
        act = 1  # env.action_space.sample()
        env.step(act)

    anim = env.render()
