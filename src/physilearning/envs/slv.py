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

        self.growth_function_flag = env_specific_params.get('growth_function_flag', 'instant')
        # self.image_sampling_type = env_specific_params.get('image_sampling_type', 'random')

        # mutant position related parameters
        self.dimension = env_specific_params.get('dimension', 2)
        self.growth_layer = env_specific_params.get('growth_layer', 150)
        self.cell_volume = env_specific_params.get('cell_volume', 2144)
        if self.dimension == 2:
            # calculate the cell area from volume
            self.cell_radius = (self.cell_volume * 3 / (4 * np.pi)) ** (1 / 3)
            self.cell_area = np.pi * self.cell_radius ** 2
            self.radius = (np.sum(self.state[0:2])/self.normalization_factor * self.cell_area / np.pi) ** (1 / 2)
        elif self.dimension == 3:
            self.radius = (np.sum(self.state[0:2])/self.normalization_factor * self.cell_volume * 3 / (4 * np.pi)) ** (1 / 3)
        else:
            raise ValueError('Dimension should be 2 or 3')

        self.mutant_radial_position = self.radius - self.mutant_distance_to_front
        self.mutant_normalized_position = self.mutant_radial_position/self.radius

        # self.drug_color = 0


    def _set_patient_specific_competition(self, patient_id):
        self.competition = [self.config['patients'][patient_id]['SLvEnv']['competition_wt'],
                            self.config['patients'][patient_id]['SLvEnv']['competition_mut']]
        self.capacity = self.config['patients'][patient_id]['SLvEnv']['carrying_capacity'] * self.normalization_factor
    def _set_patient_specific_position(self, patient_id):
        self.mutant_distance_to_front = self.config['patients'][patient_id]['SLvEnv']['mutant_distance_to_front']

    def step(self, action: int) -> Tuple[list, float, bool, bool, dict]:
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

        # check for tumor death
        if self.state[0] <= 0 and self.state[1] <= 0:
            self.state = [0, 0, 0]

        # get the reward
        rewards = Reward(self.reward_shaping_flag, normalization=np.sum(self.trajectory[0:2, 0]))
        reward += rewards.get_reward(self.state, self.time/self.max_time, self.threshold_burden)

        info = {}

        if self.observation_type == 'number':
            self.trajectory[:, self.time] = self.state
            if self.see_resistance:
                obs = self.state
            else:
                obs = [np.sum(self.state[0:2]), self.state[2]]
        # elif self.observation_type == 'image' or self.observation_type == 'multiobs':
        #     self.image = self._get_image(action)
        #     self.image_trajectory[:, :, int(self.time/self.treatment_time_step)] = self.image[0, :, :]
        #     if self.observation_type == 'image':
        #         obs = self.image
        #     elif self.observation_type == 'multiobs':
        #         obs = {'vec': self.state, 'img': self.image}
        #     else:
        #         raise NotImplementedError
        elif self.observation_type == 'mutant_position':
            self.trajectory[0:3, self.time] = self.state
            self.trajectory[3, self.time] = self.mutant_normalized_position
            self.trajectory[4, self.time] = self.radius
            if self.see_resistance:
                obs = [self.state, self.mutant_normalized_position*self.normalization_factor]
            else:
                obs = [np.sum(self.state[0:2]), self.state[2], self.mutant_normalized_position*self.normalization_factor]
        else:
            raise NotImplementedError
        terminate = self.terminate()
        truncate = self.truncate()
        # self.done = terminate or truncate
        return obs, reward, terminate, truncate, info

    def reset(self, *, seed=None, options=None):
        if self.config['env']['patient_sampling']['enable']:
            if len(self.patient_id_list) > 1:
                self._choose_new_patient()
                self._set_patient_specific_competition(self.patient_id)
                self._set_patient_specific_position(self.patient_id)
        self.time = 0
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
        if self.dimension == 2:
            self.radius = (np.sum(self.state[0:2])/self.normalization_factor * self.cell_area / np.pi) ** (1 / 2)
        elif self.dimension == 3:
            self.radius = (np.sum(self.state[0:2])/self.normalization_factor * self.cell_volume * 3 / (4 * np.pi)) ** (1 / 3)
        self.mutant_radial_position = self.radius - self.mutant_distance_to_front
        self.mutant_normalized_position = self.mutant_radial_position / self.radius

        if self.observation_type == 'number':
            self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time) + 1))
            self.trajectory[:, 0] = self.state
            if self.see_resistance:
                obs = self.state
            else:
                obs = [np.sum(self.state[0:2]), self.state[2]]

        elif self.observation_type == 'mutant_position':
            self.trajectory = np.zeros((np.shape(self.state)[0]+2, int(self.max_time) + 1))
            self.trajectory[0:3, 0] = self.state
            self.trajectory[3, 0] = self.mutant_normalized_position
            self.trajectory[4, 0] = self.radius
            if self.see_resistance:
                obs = [self.state, self.mutant_normalized_position*self.normalization_factor]
            else:
                obs = [np.sum(self.state[0:2]), self.state[2], self.mutant_normalized_position*self.normalization_factor]

        else:
            raise NotImplementedError

        return obs, {}

    # def _set_initial_mutant_positions(self):
    #     # TODO: change this to more accurate radius calculation, without image size
    #     # TODO: Make sure normalized mutant position is non-negative
    #
    #     radius = np.sqrt(np.sum(self.state[0:2])*self.cell_area/np.pi)
    #     self.mutant_radial_position
    #     ini_num_wt_to_sample = np.round(self.image_size * self.image_size * \
    #                                     self.initial_wt / (self.capacity))
    #     ini_num_mut_to_sample = np.round(self.image_size * self.image_size * \
    #                                      self.initial_mut / (self.capacity))
    #     large_radius = int(np.round(np.sqrt(ini_num_wt_to_sample + ini_num_mut_to_sample) / 3.0 * np.sqrt(2) + 1))
    #     mutant_radius = large_radius - self.mutant_distance_to_front
    #     if self.time == 0:
    #         self.angle = np.random.uniform(0, 2 * np.pi)
    #     self.mutant_x = np.round(self.image_size / 2 + mutant_radius * np.cos(self.angle))
    #     self.mutant_y = np.round(self.image_size / 2 + mutant_radius * np.sin(self.angle))
    #     radius = int(np.round(np.sqrt(ini_num_wt_to_sample) / 3.0 * np.sqrt(2) + 1))
    #     dist = radius - np.sqrt((self.mutant_x - self.image_size / 2) ** 2 + (self.mutant_y - self.image_size / 2) ** 2)
    #     self.mutant_normalized_position = dist / radius

    def _competition_function(self, dist, growth_layer) -> float:
        if dist > growth_layer:
            return (self.capacity-self.state[1])/self.state[0]
        else:
            comp = (self.capacity-self.state[1])/self.state[0]*(1-self.mutant_normalized_position)**(1/35)
            if comp < self.config['env'][self.name]['competition_wt']:
                return self.config['env'][self.name]['competition_wt']
            else:
                return comp

    def _move_mutant(self, dist, growth_layer) -> float:

        # first try deterministic move;
        # Parameters are from the fit to velocity profile
        L = 5.84
        x0 = 30.0 #80.85
        k = 0.1 #0.044
        if dist <= 0:
            self.mutant_normalized_position = 1
            self.mutant_radial_position = self.radius
        else:
            mv = L / (1 + np.exp(k*(dist-x0)))
            # print('dist: ',dist)
            # print('mv: ',mv)
            if np.random.rand() < self.mutant_normalized_position:
                self.mutant_radial_position += np.random.normal(mv, 2*mv+1) # *(3*self.cell_volume/(4*np.pi))**(1/3)
            if (self.mutant_radial_position > self.radius):
                self.mutant_radial_position = self.radius
            self.mutant_normalized_position = self.mutant_radial_position / self.radius
            if self.mutant_normalized_position > 1:
                self.mutant_normalized_position = 1
            elif self.mutant_normalized_position < 0:
                self.mutant_normalized_position = 0

        return self.mutant_normalized_position

    def grow(self, i: int, j: int, flag: str) -> float:
        # instantaneous death rate increase by drug application
        if self.dimension == 2:
            self.radius = (np.sum(self.state[0:2])/self.normalization_factor * self.cell_area / np.pi) ** (1 / 2)
        elif self.dimension == 3:
            self.radius = (np.sum(self.state[0:2])/self.normalization_factor * self.cell_volume * 3 / (4 * np.pi)) ** (1 / 3)
        #dist = (1 - self.mutant_normalized_position) * self.radius
        if self.mutant_normalized_position >= 1:
            dist = 0
        else:
            dist = self.radius - self.mutant_radial_position
        growth_layer = self.growth_layer
        self._move_mutant(dist, growth_layer)
        competition = self._competition_function(dist, growth_layer)
        self.competition[0] = competition
        new_pop_size = self.state[i] * \
                       (1 + self.growth_rate[i] *
                        (1 - (self.state[i] + self.state[j] * self.competition[j]) / self.capacity) *
                        (1 - self.death_rate_treat[i] * self.state[2]) - self.growth_rate[i] * self.death_rate[i])

        # new_pop_size += np.random.normal(0, 0.01*new_pop_size)
        if new_pop_size < 0:
            new_pop_size = 0

        return new_pop_size


if __name__ == "__main__": # pragma: no cover
    # set random seed
    np.random.seed(int(time.time()))
    env = SLvEnv.from_yaml("../../../config.yaml")
    env.reset()
    obs = [0]
    print('before loop')
    print('env_state', env.state)
    print(env.patient_id)
    print('mut normalized pos', env.mutant_normalized_position)
    print('mutant radial pos', env.mutant_radial_position)
    print('radius', env.radius)
    print('comp', env.competition)
    rad = []
    mut_rad_pos = []
    treat = []
    wt = []
    mut = []
    ini_size = env.state[0]+env.state[1]

    for i in range(250):
        if obs[0] > 1.10*ini_size:
            act = 1
        else:
            act = 0
        # act=1
        obs, rew, term, trunc, _ = env.step(act)
        rad.append(env.radius)
        mut_rad_pos.append(env.mutant_radial_position)
        treat.append(act)
        wt.append(env.state[0])
        mut.append(env.state[1])
        print('mut normalized pos',env.mutant_normalized_position)
        print('mutant radial pos', env.mutant_radial_position)
        print('radius', env.radius)
        print('comp', env.competition)
        if term or trunc:
            break
    print(i)
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(rad, label='radius')
    ax.plot(mut_rad_pos, label='mutant_radial_position')
    ax.fill_between(range(len(treat)), max(rad), max(rad) * 1.1, where=treat, color='red', alpha=0.3, label='treatment')
    ax.set_xlabel('time')
    ax.set_ylabel('radius')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(np.array(wt)+np.array(mut), label='tot')
    ax.plot(mut, label='mut')
    ax.fill_between(range(len(treat)), max(rad), max(rad) * 1.1, where=treat, color='red', alpha=0.3, label='treatment')
    ax.set_xlabel('time')
    ax.set_ylabel('number')
    ax.set_yscale('log')
    ax.legend()
    plt.show()

    #anim.save('test.mp4', fps)
