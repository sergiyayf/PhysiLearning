import numpy as np
from physilearning.envs.base_env import BaseEnv
from physilearning.reward import Reward
from typing import Tuple
import time


class SLvEnv(BaseEnv):
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
        name: str = 'SLvEnv',
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
        if self.config['env']['patient_sampling']['enable']:
            self._set_patient_specific_competition(self.patient_id)
            self._set_patient_specific_position(self.patient_id)
        else:
            self.mutant_distance_to_front = env_specific_params.get('mutant_distance_to_front', 0.0)
            self.competition = [env_specific_params.get('competition_wt', 2.),
                                env_specific_params.get('competition_mut', 1.)]

        self.growth_function_flag = env_specific_params.get('growth_function_flag', 'delayed')

        self.trajectory[:, 0] = self.state
        self.real_step_count = 0

        self.image_sampling_type = env_specific_params.get('image_sampling_type', 'random')

        self.growth_layer = env_specific_params.get('growth_layer', 10)
        self.mutant_x = 0
        self.mutant_y = 0
        self._set_initial_mutant_positions()
        self.drug_color = 0
        self.max_competition = env_specific_params.get('max_competition', 3.0)

    def _set_patient_specific_competition(self, patient_id):
        self.competition = [self.config['patients'][patient_id]['SLvEnv']['competition_wt'],
                            self.config['patients'][patient_id]['SLvEnv']['competition_mut']]
    def _set_patient_specific_position(self, patient_id):
        self.mutant_distance_to_front = self.config['patients'][patient_id]['SLvEnv']['mutant_distance_to_front']
    def _get_image(self, action: int):
        """
        Randomly sample a tumor inside of the image and return the image
        """
        # estimate the number of cells to sample
        num_wt_to_sample = np.round(self.image_size * self.image_size * \
            self.state[0] / (self.capacity))
        num_mut_to_sample = np.round(self.image_size * self.image_size * \
            self.state[1] / (self.capacity))

        mut_x = self.mutant_x
        mut_y = self.mutant_y

        # Place the resistant cells in a circle around the mutant cell
        radius = int(np.round(np.sqrt(num_mut_to_sample)/3.0*np.sqrt(2)+1))
        if mut_x-radius<0:
            min_x = 0
        else:
            min_x = mut_x-radius
        if mut_y-radius<0:
            min_y = 0
        else:
            min_y = mut_y-radius
        if mut_x+radius>self.image_size:
            max_x = self.image_size
        else:
            max_x = mut_x+radius
        if mut_y+radius>self.image_size:
            max_y = self.image_size
        else:
            max_y = mut_y+radius
        x_range = np.arange(min_x, max_x)
        y_range = np.arange(min_y, max_y)

        xx, yy = np.meshgrid(x_range, y_range)
        distances = (xx - mut_x) ** 2 + (yy - mut_y) ** 2
        mask = distances <= radius ** 2
        mut_x, mut_y = xx[mask], yy[mask]
        # make sure positions are within the image
        # mut_x, mut_y = mut_x[(mut_x >= 0) & (mut_x < self.image_size)], \
        #                 mut_y[(mut_y >= 0) & (mut_y < self.image_size)]
        if num_mut_to_sample <= 1e-2:
            mut_x, mut_y = np.array([]), np.array([])
        # remove until we have the right number
        # random_indices = np.random.randint(len(mut_x), size=int(num_mut_to_sample))
        # mut_x, mut_y = mut_x[random_indices], mut_y[random_indices]

        # put the senstitive cells inside of the circle of big radius, but not where resistant cells are
        radius = int(np.round(np.sqrt(num_wt_to_sample+num_mut_to_sample)/3.0*np.sqrt(2)+1))
        x_range = np.arange(max(self.image_size/2 - radius,0), min(self.image_size/2 + radius, self.image_size))
        y_range = np.arange(max(self.image_size/2 - radius,0), min(self.image_size/2 + radius, self.image_size))
        xx, yy = np.meshgrid(x_range, y_range)
        distances = (xx - self.image_size/2) ** 2 + (yy - self.image_size/2) ** 2
        mask = distances <= radius ** 2
        wt_x, wt_y = xx[mask], yy[mask]

        # populate the image
        # clean the image and make the new one
        if action:
            self.image = self.drug_color * np.ones((1, self.image_size, self.image_size), dtype=np.uint8)
        else:
            self.image = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)

        for x, y in zip(wt_x, wt_y):
            # color wild-type differently if they are within the growth layer
            threshold_radius = radius-self.growth_layer
            distance_to_center = np.sqrt((x-self.image_size/2)**2 + (y-self.image_size/2)**2)
            if distance_to_center < threshold_radius:
                self.image[0, int(x), int(y)] = self.wt_color
            else:
                self.image[0, int(x), int(y)] = self.wt_color-20
            #self.image[0, int(x), int(y)] = self.wt_color
        for x, y in zip(mut_x, mut_y):
            self.image[0, int(x), int(y)] = self.mut_color

        return self.image

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step in the environment that simulates tumor growth and treatment
        :param action: 0 - no treatment, 1 - treatment
        """

        # grow_tumor
        reward = 0
        self.state[2] = action
        # for t in range(0, self.treatment_time_step):
        # step time
        self.time += 1
        self.state[0] = self.grow(0, 1, self.growth_function_flag)
        self.state[1] = self.grow(1, 0, self.growth_function_flag)
        self.burden = np.sum(self.state[0:2])

        # record trajectory
        #self.state[2] = action
        self.trajectory[:, self.time] = self.state

        # check if done
        if self.state[0] <= 0 and self.state[1] <= 0:
            self.state = [0, 0, 0]

        # get the reward
        rewards = Reward(self.reward_shaping_flag, normalization=self.threshold_burden)
        reward += rewards.get_reward(self.state, self.time/self.max_time)

        info = {}

        if self.observation_type == 'number':
            if self.see_resistance:
                obs = self.state
            else:
                obs = [np.sum(self.state[0:2]), self.state[2]]
        elif self.observation_type == 'image' or self.observation_type == 'multiobs':
            self.image = self._get_image(action)
            self.image_trajectory[:, :, int(self.time/self.treatment_time_step)] = self.image[0, :, :]
            if self.observation_type == 'image':
                obs = self.image
            elif self.observation_type == 'multiobs':
                obs = {'vec': self.state, 'img': self.image}
            else:
                raise NotImplementedError
        elif self.observation_type == 'mutant_position':
            if self.see_resistance:
                obs = [self.state, self.mutant_normalized_position]
            else:
                obs = [np.sum(self.state[0:2]), self.state[2], self.mutant_normalized_position]
        else:
            raise NotImplementedError
        terminate = self.terminate()
        truncate = self.truncate()
        self.done = terminate or truncate
        return obs, reward, terminate, truncate, info

    def reset(self, *, seed=None, options=None):
        self.real_step_count += 1
        if self.config['env']['patient_sampling']['enable']:
            if len(self.patient_id_list) > 1:
                self._choose_new_patient()
                self._set_patient_specific_competition(self.patient_id)
                self._set_patient_specific_position(self.patient_id)
        self.time = 0
        self._set_initial_mutant_positions()
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


        self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time)+1))
        self.trajectory[:, 0] = self.state

        if self.observation_type == 'number':
            if self.see_resistance:
                obs = self.state
            else:
                obs = [np.sum(self.state[0:2]), self.state[2]]
        elif self.observation_type == 'image' or self.observation_type == 'multiobs':
            self.image = self._get_image(self.initial_drug)
            self.image_trajectory = np.zeros(
                (self.image_size, self.image_size, int(self.max_time / self.treatment_time_step) + 1))
            self.image_trajectory[:, :, 0] = self.image[0, :, :]
            if self.observation_type == 'image':
                obs = self.image
            elif self.observation_type == 'multiobs':
                obs = {'vec': self.state, 'img': self.image}
        elif self.observation_type == 'mutant_position':
            if self.see_resistance:
                obs = [self.state, self.mutant_normalized_position]
            else:
                obs = [np.sum(self.state[0:2]), self.state[2], self.mutant_normalized_position]

        else:
            raise NotImplementedError

        return obs, {}

    def _set_initial_mutant_positions(self):
        ini_num_wt_to_sample = np.round(self.image_size * self.image_size * \
                                        self.initial_wt / (self.capacity))
        ini_num_mut_to_sample = np.round(self.image_size * self.image_size * \
                                         self.initial_mut / (self.capacity))
        large_radius = int(np.round(np.sqrt(ini_num_wt_to_sample + ini_num_mut_to_sample) / 3.0 * np.sqrt(2) + 1))
        mutant_radius = large_radius - self.mutant_distance_to_front
        if self.time == 0:
            self.angle = np.random.uniform(0, 2 * np.pi)
        self.mutant_x = np.round(self.image_size / 2 + mutant_radius * np.cos(self.angle))
        self.mutant_y = np.round(self.image_size / 2 + mutant_radius * np.sin(self.angle))
        radius = int(np.round(np.sqrt(ini_num_wt_to_sample) / 3.0 * np.sqrt(2) + 1))
        dist = radius - np.sqrt((self.mutant_x - self.image_size / 2) ** 2 + (self.mutant_y - self.image_size / 2) ** 2)
        self.mutant_normalized_position = dist / radius

    def _competition_function(self, dist, growth_layer) -> float:
        if dist > growth_layer:
            return (self.capacity-self.state[1])/self.state[0]
        else:
            return (self.capacity-self.state[1])/self.state[0]*dist/growth_layer

    def _move_mutant(self, dist, growth_layer) -> float:

        if dist > growth_layer:
            mv = 0
        elif dist > 0:
            mv = 1-dist/growth_layer
        else:
            mv = 0

        if np.random.uniform() < mv:
            # move mutant on the grid radially outward
            center = [self.image_size / 2, self.image_size / 2]
            # calculate the angle between the mutant and the center
            angle = np.arctan2(self.mutant_y - center[1], self.mutant_x - center[0])
            # calculate the new position of the mutant
            if np.random.uniform() < abs(np.cos(angle)):
                self.mutant_x += np.sign(np.cos(angle))
            if np.random.uniform() < abs(np.sin(angle)):
                self.mutant_y += np.sign(np.sin(angle))

    def grow(self, i: int, j: int, flag: str) -> float:

        # instantaneous death rate increase by drug application
        num_wt_to_sample = np.round(self.image_size * self.image_size * \
                                        self.state[0] / (self.capacity))

        radius = int(np.round(np.sqrt(num_wt_to_sample) / 3.0 * np.sqrt(2) + 1))
        dist = radius - np.sqrt((self.mutant_x - self.image_size / 2) ** 2 + (self.mutant_y - self.image_size / 2) ** 2)
        self.mutant_normalized_position = dist / radius
        growth_layer = self.growth_layer
        self._move_mutant(dist, growth_layer)
        competition = self._competition_function(dist, growth_layer)
        self.competition = [competition, 0]

        new_pop_size = self.state[i] * \
                       (1 + self.growth_rate[i] *
                        (1 - (self.state[i] + self.state[j] * self.competition[j]) / self.capacity) -
                        self.death_rate[i] -
                        self.death_rate_treat[i] * self.state[2])
        if new_pop_size < 0:
            new_pop_size = 0

        return new_pop_size


if __name__ == "__main__": # pragma: no cover
    # set random seed
    np.random.seed(int(time.time()))
    env = SLvEnv.from_yaml("../../../config.yaml")
    env.reset()
    grid = env.image
    obs = [0]
    print('before loop')
    print(env.patient_id)
    for i in range(150):
        if i%2 == 0:
            act = 1
        else:
            act = 0
        obs, rew, term, trunc, _ = env.step(act)
        #print(env.mutant_normalized_position)
        if term or trunc:
            break
    print(i)
    anim = env.render()
    env.reset()
    grid = env.image
    obs = [0]
    print('before loop')
    print(env.patient_id)
    for i in range(150):
        if i % 2 == 0:
            act = 1
        else:
            act = 0
        #print(env.mutant_normalized_position)
        obs, rew, term, trunc, _ = env.step(act)
        if term or trunc:
            break
    print(i)
    anim = env.render()
    env.reset()
    grid = env.image
    obs = [0]
    print('before loop')
    print(env.patient_id)
    for i in range(150):
        if i % 2 == 0:
            act = 1
        else:
            act = 0
        # print(env.mutant_normalized_position)
        obs, rew, term, trunc, _ = env.step(act)
        if term or trunc:
            break
    print(i)
    anim = env.render()
    #anim.save('test.mp4', fps)
