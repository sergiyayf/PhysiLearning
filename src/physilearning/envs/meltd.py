import numpy as np
from physilearning.envs.base_env import BaseEnv
from physilearning.reward import Reward
from typing import Tuple
import time


class MeltdEnv(BaseEnv):
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
        # if self.normalize:
        #     self.death_rate_treat[0] *= self.normalization_factor
        #     self.death_rate_treat[1] *= self.normalization_factor


        # 1 - wt, 2 - resistant
        self.treat = 0
        if self.config['env']['patient_sampling']['enable']:
            self._set_patient_specific_position(self.patient_id)
        else:
            self.mutant_distance_to_front = env_specific_params.get('mutant_distance_to_front', 0.0)

        self.growth_function_flag = env_specific_params.get('growth_function_flag', 'instant')
        # self.image_sampling_type = env_specific_params.get('image_sampling_type', 'random')

        # mutant position related parameters
        self.dimension = env_specific_params.get('dimension', 2)
        self.growth_layer = env_specific_params.get('growth_layer', 150)
        self.cell_area = env_specific_params.get('cell_area', 2144)

        if self.dimension == 2:
            # calculate the cell area from volume
            self.cell_radius = (self.cell_area/np.pi) ** (1 / 2)
            self.cell_volume = (self.cell_radius ** 3) * np.pi * 4 / 3
            self.radius = (np.sum(self.state[0:2])/self.normalization_factor * self.cell_area / np.pi) ** (1 / 2)
        elif self.dimension == 3:
            self.radius = (np.sum(self.state[0:2])/self.normalization_factor * self.cell_volume * 3 / (4 * np.pi)) ** (1 / 3)
        else:
            raise ValueError('Dimension should be 2 or 3')

        self.mutant_radial_position = self.radius - self.mutant_distance_to_front
        self.mutant_normalized_position = self.mutant_radial_position/self.radius


    def _set_patient_specific_position(self, patient_id):
        self.mutant_distance_to_front = self.config['patients'][patient_id]['MeltdEnv']['mutant_distance_to_front']

    def step(self, action: int) -> Tuple[list, float, bool, bool, dict]:
        """
        Step in the environment that simulates tumor growth and treatment
        :param action: 0 - no treatment, 1 - treatment
        """

        # grow_tumor
        reward = 0
        self.state[2] = action
        for t in range(0, self.treatment_time_step):
            # step time
            self.time += 1
            self.state[0] = self.grow(0, 1, self.growth_function_flag)
            self.state[1] = self.grow(1, 0, self.growth_function_flag)
            self.burden = np.sum(self.state[0:2])

            # check for tumor death
            if self.state[0] <= 0 and self.state[1] <= 0:
                self.state = [0, 0, 0]

            info = {}

            if self.observation_type == 'number':
                self.trajectory[:, self.time] = self.state

                if self.see_resistance:
                    obs = self.state[0:2]
                else:
                    obs = [np.sum(self.state[0:2])]

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

            # get the reward
            rewards = Reward(self.reward_shaping_flag, normalization=np.sum(self.trajectory[0:2, 0]))
            if self.reward_shaping_flag == 'tendayaverage':
                reward += rewards.tendayaverage(self.trajectory, self.time)
            else:
                reward += rewards.get_reward(self.state, self.time / self.max_time, self.threshold_burden)
        terminate = self.terminate()
        truncate = self.truncate()
        # self.done = terminate or truncate
        return obs, reward, terminate, truncate, info

    def reset(self, *, seed=None, options=None):
        if self.config['env']['patient_sampling']['enable']:
            if len(self.patient_id_list) > 1:
                self._choose_new_patient()
                self._set_patient_specific_position(self.patient_id)
        else:
            self.mutant_distance_to_front = np.random.uniform(0, 1300)
        self.time = 0
        if self.wt_random:
            self.initial_wt = \
                np.random.random_integers(low=0, high=int(self.max_tumor_size), size=1)[0]
            if self.normalize:
                self.initial_wt = self.initial_wt*self.normalization_factor
        if self.mut_random:
            self.initial_mut = \
                np.random.random_integers(low=1, high=15, size=1)[0]
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
                obs = self.state[0:2]
            else:
                obs = [np.sum(self.state[0:2])]

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
        for tt in [0, 1]:
            self.time += 1
            self.state[0] = self.grow(0, 1, self.growth_function_flag)
            self.state[1] = self.grow(1, 0, self.growth_function_flag)
            self.burden = np.sum(self.state[0:2])
            # record trajectory
            # self.state[2] = action
            self.trajectory[:, self.time] = self.state
        self.threshold_burden = self.max_tumor_size * (self.state[0]+self.state[1])
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
            # 2D move
            a = -0.39299
            b = 172.3
            c = 25.48
            mv = (a*dist+b-c)*np.heaviside(a*dist+b-c, 1) + c

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

        if self.mutant_normalized_position >= 1:
            dist = 0
        else:
            dist = self.radius - self.mutant_radial_position
        growth_layer = self.growth_layer
        self._move_mutant(dist, growth_layer)

        if flag == 'instant' or flag == 'instant_with_noise':
            self.treat = self.state[2]
        elif flag == 'delayed' or flag == 'delayed_with_noise':

            if self.state[2] == 0:
                if self.time > 1 and (self.trajectory[2, self.time - 1] == 1):
                    self.treat = 1
                elif self.time > 2 and (self.trajectory[2, self.time - 2] == 1):
                    self.treat = 1
                elif self.time > 3 and (self.trajectory[2, self.time - 3] == 1):
                    self.treat = 1
                elif self.time > 4 and (self.trajectory[2, self.time - 4] == 1):
                    self.treat = 1
                elif self.time > 5 and (self.trajectory[2, self.time - 5] == 1):
                    self.treat = 1
                else:
                    self.treat = 0
            elif self.state[2] == 1:
                if self.time in [0, 1, 2, 3]:
                    self.treat = 0

                elif (self.trajectory[2, self.time - 1] == 0):
                    self.treat = 0
                elif (self.trajectory[2, self.time - 2] == 0):
                    self.treat = 0
                elif (self.trajectory[2, self.time - 3] == 0):
                    self.treat = 0
                else:
                    self.treat = 1

        if i == 0:
            new_pop_size = self.state[i] * (1 + self.growth_rate[i]*(1 - self.death_rate_treat[i] * self.treat))

        else:
            #
            if self.state[1] > 0.1*(self.initial_wt+self.initial_mut):
                # calculate radius estimate of purely state 1 colony
                red_colony_rad = (self.state[1]/self.normalization_factor * self.cell_area / np.pi) ** (1 / 2)
                dist += red_colony_rad
                a = -3.47226633e-04
                b = 5.11816467e-01
                growth_rate = (a * dist + b) * np.heaviside(a * dist + b, 1) / 2
                if growth_rate < self.growth_rate[i]:
                    growth_rate = self.growth_rate[i]
            else:

                a = -3.47226633e-04
                b = 5.11816467e-01
                growth_rate = (a * dist + b) * np.heaviside(a * dist + b, 1) / 2

            new_pop_size = self.state[i] * (1+growth_rate)
        if new_pop_size < 10 * self.normalization_factor and self.death_rate_treat[i] * self.treat > 0:
            new_pop_size = 0

        elif flag == 'instant_with_noise' or flag == 'delayed_with_noise':
            # rand = truncnorm(loc=0, scale=0.00528*new_pop_size, a=-0.02/0.00528, b=0.02/0.00528).rvs()
            rand = np.random.normal(0, 0.01 * new_pop_size, 1)[0]
            if np.abs(rand) > 0.1 * new_pop_size:
                rand = 0.1 * new_pop_size * np.sign(rand)
            new_pop_size += rand
            if new_pop_size < 10 * self.normalization_factor and self.death_rate_treat[i] * self.treat > 0:
                new_pop_size = 0

        if new_pop_size < 0:
            new_pop_size = 0

        return new_pop_size


if __name__ == "__main__": # pragma: no cover
    # for debugging the environment and checking its behavior under various treatments without job submission
    # set random seed
    np.random.seed(int(time.time()))
    env = MeltdEnv.from_yaml("../../../config.yaml")
    env.reset()
    obs = [env.state[0]+env.state[1]]
    rad = []
    mut_rad_pos = []
    treat = []
    wt = []
    mut = []
    ini_size = env.state[0]+env.state[1]
    maxtime = 150
    for i in range(maxtime):
        print(env.radius)
        # on 2x off treatment

        if obs[0] > 1.85*ini_size and env.state[2] == 0:
            act = 1
        else:
            act = 0
        act = 1
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
    ax.set_xlim(0, maxtime)
    ax.set_title('Radius and mutant radial position')

    fig, ax = plt.subplots()
    time = [i for i in range(len(env.trajectory[0, :]))]
    # rollback treatment
    treat = env.trajectory[2, :]
    treat = np.where(treat == 0, np.roll(treat, -1), treat)
    ax.plot(time, env.trajectory[0, :]/(ini_size), label='wt', color='green')
    ax.plot(time, env.trajectory[1, :]/(ini_size), label='mut', color='red')
    ax.plot(time, env.trajectory[0, :]/(ini_size)+env.trajectory[1, :]/(ini_size), label='tot', color='k')
    ax.fill_between(range(len(treat)), 3, 4, where=treat, color='orange', alpha=0.3, label='treatment')
    ax.set_title('All resolution simulation data')

    # plot only dayly data
    fig, ax = plt.subplots()
    sens = env.trajectory[0, ::2]/(ini_size)
    res = env.trajectory[1, ::2]/(ini_size)
    tot = sens + res
    time = [i for i in range(len(sens))]
    ax.plot(time, sens, label='wt', color='green')
    ax.plot(time, res, label='mut', color='red')
    ax.plot(time, tot, label='tot', color='k')
    ax.fill_between(range(len(treat[::2])), 3, 4, where=treat[::2], color='orange', alpha=0.3, label='treatment')
    ax.legend()
    ax.set_title('Daily data')

    # plot every second day
    fig, ax = plt.subplots()
    sens = env.trajectory[0, ::4]/(ini_size)
    res = env.trajectory[1, ::4]/(ini_size)
    tot = sens + res
    time = [i for i in range(len(sens))]
    ax.plot(time, sens, label='wt', color='green')
    ax.plot(time, res, label='mut', color='red')
    ax.plot(time, tot, label='tot', color='k')
    ax.fill_between(range(len(treat[::4])), 3, 4, where=treat[::4], color='orange', alpha=0.3, label='treatment')
    ax.legend()
    ax.set_title('Every second day data')
    plt.show()
    #anim.save('test.mp4', fps)