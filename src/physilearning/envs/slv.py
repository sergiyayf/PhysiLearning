import numpy as np
from physilearning.envs.base_env import BaseEnv
from physilearning.reward import Reward
from typing import Tuple
import time
from scipy.stats import truncnorm


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
        see_resistance: bool = False,
        see_prev_action: bool = False,
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
            initial_wt = np.random.randint(low=1000, high=3000, size=1)[0]
        if self.mut_random:
            initial_mut = np.random.randint(low=0, high=20, size=1)[0]
        super().__init__(config=config, name=name, observation_type=observation_type, action_type=action_type,
                         max_tumor_size=max_tumor_size, max_time=max_time, initial_wt=initial_wt,
                         initial_mut=initial_mut, growth_rate_wt=growth_rate_wt, growth_rate_mut=growth_rate_mut,
                         death_rate_wt=death_rate_wt, death_rate_mut=death_rate_mut,
                         treat_death_rate_wt=treat_death_rate_wt, treat_death_rate_mut=treat_death_rate_mut,
                         treatment_time_step=treatment_time_step, reward_shaping_flag=reward_shaping_flag,
                         normalize=normalize, normalize_to=normalize_to, image_size=image_size, patient_id=patient_id,
                         see_resistance=see_resistance, see_prev_action=see_prev_action, env_specific_params=env_specific_params,
                         )

        self.capacity_non_normalized = env_specific_params.get('carrying_capacity', 6500)
        # Normalizazion
        if self.normalize:
            self.capacity = env_specific_params.get('carrying_capacity', 6500) \
                            * self.normalization_factor
        else:
            self.capacity = env_specific_params.get('carrying_capacity', 6500)

        self.growth_function_flag = env_specific_params.get('growth_function_flag', 'delayed')
        self.trajectory[:, 0] = self.state

        if self.growth_function_flag == 'instant_fixed_treat' or self.growth_function_flag == 'instant_fixed_treat_with_noise':
            self.death_rate_treat[0] *= self.normalization_factor
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
        self.growth_fit = env_specific_params.get('growth_fit', 'exp')
        if self.dimension == 2:
            # calculate the cell area from volume
            self.cell_radius = (self.cell_volume * 3 / (4 * np.pi)) ** (1 / 3)
            self.cell_area = np.pi * self.cell_radius ** 2
            self.radius = (np.sum(self.state[0:2])/self.normalization_factor * self.cell_area / np.pi) ** (1 / 2)
        elif self.dimension == 3:
            self.radius = (np.sum(self.state[0:2])/self.normalization_factor * self.cell_volume * 3 / (4 * np.pi)) ** (1 / 3)
        else:
            raise ValueError('Dimension should be 2 or 3')

        if isinstance(self.mutant_distance_to_front, str):
            self.random_distance = True
            self.mutant_distance_to_front = np.random.randint(low=0, high=self.radius*0.5, size=1)[0]
        else:
            self.random_distance = False
        self.mutant_radial_position = self.radius - self.mutant_distance_to_front
        self.mutant_normalized_position = self.mutant_radial_position/self.radius


    def _set_patient_specific_competition(self, patient_id):
        self.competition = [self.config['patients'][patient_id]['SLvEnv']['competition_wt'],
                            self.config['patients'][patient_id]['SLvEnv']['competition_mut']]
        self.capacity = self.config['patients'][patient_id]['SLvEnv']['carrying_capacity'] * self.normalization_factor
    def _set_patient_specific_position(self, patient_id):
        self.mutant_distance_to_front = self.config['patients'][patient_id]['SLvEnv']['mutant_distance_to_front']

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step in the environment that simulates tumor growth and treatment
        :param action: 0 - no treatment, 1 - treatment
        """

        # grow_tumor
        reward = 0
        for t in range(0, self.treatment_time_step):
            # step time
            self.time += 1
            self.state[2] = action
            self.state[0] = self.grow(0, 1, self.growth_function_flag)
            self.state[1] = self.grow(1, 0, self.growth_function_flag)
            self.burden = np.sum(self.state[0:2])

            self.trajectory[:, self.time] = self.state
            # check if done
            if self.state[0] <= 0 and self.state[1] <= 0:
                self.state = [0, 0, 0]

            # get the reward
            reward += self.get_reward()

        info = {}
        if self.observation_type == 'number':
            if self.see_resistance:
                obs = self.state[0:2]
            else:
                obs = [np.sum(self.state[0:2])]
            if self.see_prev_action:
                obs = np.append(obs, action)
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
        self.done = terminate or truncate

        return obs, reward, terminate, truncate, info


    def reset(self, *, seed=None, options=None):
        if self.config['env']['patient_sampling']['enable']:
            if len(self.patient_id_list) > 1:
                self._choose_new_patient()
                self._set_patient_specific_competition(self.patient_id)

        self.randomize_params()
        if self.normalize:
            keys = self.random_params.keys()
            if 'initial_wt' in keys or 'initial_mut' in keys:
                self.normalization_factor = self.normalize_to / (self.initial_wt + self.initial_mut)
                self.initial_wt *= self.normalization_factor
                self.initial_mut *= self.normalization_factor
                self.capacity = self.capacity_non_normalized * self.normalization_factor


        self.state = [self.initial_wt, self.initial_mut, self.initial_drug]
        self.time = 0
        self.time_on_treatment = 0
        self.current_rew = 0
        self.done = False
        self.mutant_distance_to_front = self.config['env']['SLvEnv']['mutant_distance_to_front']
        if self.dimension == 2:
            # calculate the cell area from volume
            self.cell_radius = (self.cell_volume * 3 / (4 * np.pi)) ** (1 / 3)
            self.cell_area = np.pi * self.cell_radius ** 2
            self.radius = (np.sum(self.state[0:2])/self.normalization_factor * self.cell_area / np.pi) ** (1 / 2)
        elif self.dimension == 3:
            self.radius = (np.sum(self.state[0:2])/self.normalization_factor * self.cell_volume * 3 / (4 * np.pi)) ** (1 / 3)
        self.mutant_radial_position = self.radius - self.mutant_distance_to_front
        self.mutant_normalized_position = self.mutant_radial_position / self.radius

        if self.observation_type == 'number':
            self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time) + 1))
            self.trajectory[:, 0] = self.state
            if self.see_resistance:
                obs = self.state[0:2]
            else:
                obs = [np.sum(self.state[0:2])]
            if self.see_prev_action:
                obs = np.append(obs, 0)
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

    def _move_mutant(self, dist, growth_layer) -> float:
        """
        Move the mutant cell towards the front of the tumor depending on the distance from the front
        """

        if dist <= 0:
            self.mutant_normalized_position = 1
            self.mutant_radial_position = self.radius
        else:
            #mv = L / (1 + np.exp(k*(dist-x0)))
            #mv = (-0.0565 * dist + 4.76)*np.heaviside(-0.0565 * dist + 4.76, 1)
            #mv = (-0.065 * dist + 5.007) * np.heaviside(-0.065 * dist + 5.007, 1)
            a = 148 #139
            b = 2.57e-4 #6.26e-4
            mv = (b*(a-dist)**2) * np.heaviside((a-dist), 1)

            if True: #np.random.rand() < self.mutant_normalized_position:
                self.mutant_radial_position += mv #np.random.normal(mv,3*mv+3) # *(3*self.cell_volume/(4*np.pi))**(1/3)
                print(mv)
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

        if self.mutant_normalized_position >= 1:
            dist = 0
        else:
            dist = self.radius - self.mutant_radial_position
        growth_layer = self.growth_layer
        self._move_mutant(dist, growth_layer)

        if i == 0:
            # new_pop_size = self.state[i] * \
            #                (1 + self.growth_rate[i] *
            #                 (1 - (self.state[i] + self.state[j] * self.competition[j]) / self.capacity) *
            #                 (1 - self.death_rate_treat[i] * self.state[2]) - self.growth_rate[i] * self.death_rate[i])
            new_pop_size = self.state[i] * \
                           (1 + self.growth_rate[i] *
                            (1 - (self.state[i] + self.state[j] * self.competition[j]) / self.capacity) -
                            self.growth_rate[i] * self.death_rate[i]) - self.death_rate_treat[i] * self.state[2]
        else:
            #
            if self.state[0] > self.state[1]:
                # fitted growth rate minus base death rate
                if self.growth_fit == 'exp':
                    growth_rate = 0.139*np.exp(-0.0173*dist) - 0.033
                elif self.growth_fit == 'linear':
                    growth_rate = (-0.000998 * dist + 0.1227) * np.heaviside(-0.000998 * dist + 0.1227, 1) - 0.033
                elif self.growth_fit == 'quadratic':
                    a = 215 #213 #220
                    b = 2.29e-6 #2.17e-6 #2.38e-6
                    growth_rate = (b*(a-dist)**2) * np.heaviside((a-dist), 1) - 0.001
                else:
                    print('Specified growth fit is not found, using position independent growth rate')
                    growth_rate = self.growth_rate[i]
            else:
                growth_rate = self.growth_rate[i]

            new_pop_size = self.state[i] * (1+growth_rate)
        if new_pop_size < 10 * self.normalization_factor and self.death_rate_treat[i] * self.state[2] > 0:
            new_pop_size = 0

        if flag == 'instant':
            pass
        elif 'with_noise' in flag:
            # rand = truncnorm(loc=0, scale=0.00528*new_pop_size, a=-0.02/0.00528, b=0.02/0.00528).rvs()
            if new_pop_size < 10 * self.normalization_factor and self.death_rate_treat[i] * self.state[2] > 0:
                new_pop_size = 0
            rand = np.random.normal(0, 0.01 * new_pop_size, 1)[0]
            if np.abs(rand) > 0.05 * new_pop_size:
                rand = 0.05 * new_pop_size * np.sign(rand)
            new_pop_size += rand
            if new_pop_size < 10 * self.normalization_factor and self.death_rate_treat[i] * self.state[2] > 0:
                new_pop_size = 0

        if new_pop_size < 0:
            new_pop_size = 0

        return new_pop_size


if __name__ == "__main__": # pragma: no cover
    # for debugging the environment and checking its behavior under various treatments without job submission
    # set random seed
    np.random.seed(int(time.time()))
    env = SLvEnv.from_yaml("../../../config.yaml")
    env.reset()
    rad = []
    mut_rad_pos = []
    treat = []
    wt = []
    mut = []
    ini_size = env.state[0]+env.state[1]
    wt.append(env.state[0])
    mut.append(env.state[1])
    treat.append(0)
    obs = [env.state[0]+ env.state[1]]
    rad.append(env.radius)
    mut_rad_pos.append(env.mutant_radial_position)

    for i in range(600):
        if obs[0] >= 1.0*ini_size:
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
        if term or trunc:
            break

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    norm = rad[0]
    rad = rad / norm
    mut_rad_pos = mut_rad_pos / norm
    mut0 = mut[0]
    wt0 = wt[0]
    mut = np.array(mut)/(mut0+wt0)
    wt = np.array(wt)/(mut0+wt0)
    ax.plot(rad, label='radius')
    ax.plot(mut_rad_pos, label='mutant_radial_position')
    ax.plot(mut, label='mut', color='red')
    ax.fill_between(range(len(treat)), max(rad), max(rad) * 1.1, where=treat, color='orange', alpha=0.3, label='treatment')
    ax.set_xlabel('time')
    ax.set_ylabel('radius')
    ax.legend()
    ax.set_xlim(0, 100)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(wt, label='wt')
    ax.plot(mut, label='mut')
    ax.fill_between(range(len(treat)), max(rad), max(rad) * 1.1, where=treat, color='orange', alpha=0.3, label='treatment')
    ax.set_xlabel('time')
    ax.set_ylabel('number')
    ax.set_yscale('linear')
    ax.legend()
    ax.set_xlim(0, 100)
    plt.show()

    #anim.save('test.mp4', fps)
