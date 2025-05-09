import numpy as np
from physilearning.envs.base_env import BaseEnv
from physilearning.reward import Reward
from typing import Tuple
import time
from scipy.stats import truncnorm


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
        reward_shaping_flag: str = 'ttp',
        normalize:  bool = 1,
        normalize_to: float = 1000,
        image_size: int = 84,
        patient_id: int | list = 0,
        env_specific_params: dict = {},
        **kwargs,
    ) -> None:

        super().__init__(config=config, name=name, observation_type=observation_type, action_type=action_type,
                         max_tumor_size=max_tumor_size, max_time=max_time, initial_wt=initial_wt,
                         initial_mut=initial_mut, growth_rate_wt=growth_rate_wt, growth_rate_mut=growth_rate_mut,
                         death_rate_wt=death_rate_wt, death_rate_mut=death_rate_mut,
                         treat_death_rate_wt=treat_death_rate_wt, treat_death_rate_mut=treat_death_rate_mut,
                         treatment_time_step=treatment_time_step, reward_shaping_flag=reward_shaping_flag,
                         normalize=normalize, normalize_to=normalize_to, image_size=image_size, patient_id=patient_id,
                         )

        self.capacity_non_normalized = env_specific_params.get('carrying_capacity', 6500)
        # Normalizazion
        if self.normalize:
            self.capacity = env_specific_params.get('carrying_capacity', 6500) \
                            * self.normalization_factor
        else:
            self.capacity = env_specific_params.get('carrying_capacity', 6500)

        # 1 - wt, 2 - resistant
        if self.config['env']['patient_sampling']['enable']:
            self._set_patient_specific_competition(self.patient_id)
        else:
            self.competition = [env_specific_params.get('competition_wt', 2.),
                                env_specific_params.get('competition_mut', 1.)]

        self.growth_function_flag = env_specific_params.get('growth_function_flag', 'delayed')

        self.trajectory[:, 0] = self.state

        self.image_sampling_type = env_specific_params.get('image_sampling_type', 'random')

        if self.growth_function_flag == 'instant_fixed_treat' or self.growth_function_flag == 'instant_fixed_treat_with_noise':
            self.death_rate_treat[0] *= self.normalization_factor

        self.mtd_rew = 0
        self.current_rew = 0
        self.k = env_specific_params.get('k', 0.1)
        self.t0 = env_specific_params.get('t0', 100)
        self.time_on_treatment = 0
        self.end_time = 120

    def _set_patient_specific_competition(self, patient_id):
        self.competition = [self.config['patients'][patient_id]['LvEnv']['competition_wt'],
                            self.config['patients'][patient_id]['LvEnv']['competition_mut']]
        self.capacity = self.config['patients'][patient_id]['LvEnv']['carrying_capacity']*self.normalization_factor

    def _get_image(self, action: int):
        """
        Randomly sample a tumor inside of the image and return the image
        """
        # estimate the number of cells to sample
        num_wt_to_sample = np.round(self.image_size * self.image_size * \
            self.state[0] / (self.capacity))
        num_mut_to_sample = np.round(self.image_size * self.image_size * \
            self.state[1] / (self.capacity))

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

            radius = int(np.round(np.sqrt(num_wt_to_sample+num_mut_to_sample)/2.6*np.sqrt(2)+1))

            x_range = np.arange(self.image_size/2 - radius, self.image_size/2 + radius + 1)
            y_range = np.arange(self.image_size/2 - radius, self.image_size/2 + radius + 1)
            xx, yy = np.meshgrid(x_range, y_range)
            distances = (xx - self.image_size/2) ** 2 + (yy - self.image_size/2) ** 2

            # Create a mask for the coordinates that fall within the circular region
            mask = distances <= radius ** 2
            wt_x, wt_y = xx[mask], yy[mask]

            # remove some cells until we have the right number
            random_indices = np.random.randint(len(wt_x), size=int(num_wt_to_sample))
            wt_x, wt_y = wt_x[random_indices], wt_y[random_indices]

            # place resistant cells in the remaining space inside the circle
            mut_x, mut_y = xx[mask], yy[mask]
            # remove the indices that are already occupied by sensitive cells
            mut_x, mut_y = np.delete(mut_x, random_indices), np.delete(mut_y, random_indices)

            random_indices = np.random.randint(len(mut_x), size=int(num_mut_to_sample))
            mut_x, mut_y = mut_x[random_indices], mut_y[random_indices]

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

            if action:
                self.time_on_treatment += 1
            else:
                self.time_on_treatment = 0

            # record trajectory
            #self.state[2] = action
            #self.trajectory[:, self.time] = self.state

            # check if done
            # if self.state[0] <= 0 and self.state[1] <= 0:
            #     self.state = [0, 0, 0]

            # get the reward
            reward += self.get_reward()

        self.current_rew += reward

        info = {}

        if self.observation_type == 'number':
            if self.see_resistance:
                obs = self.state[0:2]
            else:
                obs = [np.sum(self.state[0:2])]

        terminate = self.terminate()
        truncate = self.truncate()
        self.done = terminate or truncate

        if self.reward_shaping_flag == 'mtd_compare':
            if self.done:
                print('MTD rew: ', self.mtd_rew)
                print('Current rew: ', self.current_rew)
                reward = (self.current_rew/self.mtd_rew-1)*100
            else:
                reward = 0

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

        self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time)+1))
        self.trajectory[:, 0] = self.state

        if self.observation_type == 'number':
            if self.see_resistance:
                obs = self.state[0:2]
            else:
                obs = [np.sum(self.state[0:2])]
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

        # do day zero without treatment
        # for tt in [0, 1]:
        #     self.time += 1
        #     self.state[0] = self.grow(0, 1, self.growth_function_flag)
        #     self.state[1] = self.grow(1, 0, self.growth_function_flag)
        #     self.burden = np.sum(self.state[0:2])
        #     # record trajectory
        #     # self.state[2] = action
        #     self.trajectory[:, self.time] = self.state
        self.threshold_burden = self.max_tumor_size * (self.state[0]+self.state[1])

        if self.reward_shaping_flag == 'mtd_compare':
            # backup parameters
            backup_state = self.state.copy()
            backup_time = self.time
            backup_trajectory = self.trajectory.copy()
            self.mtd_rew = self.run_mtd()
            # reset params back to original
            self.state = backup_state
            self.time = backup_time
            self.trajectory = backup_trajectory

        return obs, {}

    def run_mtd(self):
        rew = 0
        while not self.done:
            for tt in range(0, self.treatment_time_step):
                self.time += 1
                self.state[2] = 1
                self.state[0] = self.grow(0, 1, self.growth_function_flag)
                self.state[1] = self.grow(1, 0, self.growth_function_flag)
                self.trajectory[:, self.time] = self.state
                rew += Reward(self.reward_shaping_flag, normalization=np.sum(self.trajectory[0:2, 0])).tendayaverage(self.trajectory, self.time)
            terminate = self.terminate()
            truncate = self.truncate()
            self.done = terminate or truncate
        self.done = False
        if rew == 0:
            rew = 100
        return rew

    def grow(self, i: int, j: int, flag: str) -> float:

        # instantaneous death rate increase by drug application
        if flag == 'instant' or flag == 'instant_with_noise':
            new_pop_size = self.state[i] * \
                           (1 + self.growth_rate[i] *
                            (1 - (self.state[i] + self.state[j] * self.competition[j]) / self.capacity)
                            - self.growth_rate[i] * self.death_rate[i]) - self.death_rate_treat[i] * self.state[2]
        elif flag == 'instant_fixed_treat' or flag == 'instant_fixed_treat_with_noise':

            new_pop_size = self.state[i] * \
                           (1 + self.growth_rate[i] *
                            (1 - (self.state[i] + self.state[j] * self.competition[j]) / self.capacity) -
                            self.growth_rate[i] * self.death_rate[i]) - self.death_rate_treat[i] * self.state[2]
        elif flag == '3D':
            new_pop_size = self.state[i] * \
                           (1 + self.growth_rate[i] *
                            (1 - (self.state[i] + self.state[j] * self.competition[j]) / self.capacity) -
                            self.growth_rate[i] * self.death_rate[i]) -self.state[2]*self.death_rate_treat[i]*self.state[i]/(1+np.exp(-self.k*(self.time_on_treatment-self.t0)))
        # one time step delay in treatment effect
        elif flag == 'delayed' or flag == 'delayed_with_noise':
            treat = self.state[2]
            if self.state[2] == 0:
                if self.time > 1 and (self.trajectory[2, self.time - 1] == 1):
                    treat = 1
                elif self.time > 2 and (self.trajectory[2, self.time - 2] == 1):
                    treat = 1
                elif self.time > 3 and (self.trajectory[2, self.time - 3] == 1):
                    treat = 1
                elif self.time > 4 and (self.trajectory[2, self.time - 4] == 1):
                    treat = 1
                elif self.time > 5 and (self.trajectory[2, self.time - 5] == 1):
                    treat = 1
                else:
                    treat = 0
            elif self.state[2] == 1:
                if self.time in [0, 1, 2, 3]:
                    treat = 0
                elif (self.trajectory[2, self.time - 1] == 0):
                    treat = 0
                elif (self.trajectory[2, self.time - 2] == 0):
                    treat = 0
                elif (self.trajectory[2, self.time - 3] == 0):
                    treat = 0
                else:
                    treat = 1

            new_pop_size = self.state[i] * (1 + self.growth_rate[i] *
                (1 - (self.state[i] + self.state[j] * self.competition[j]) / self.capacity)*(1 - self.death_rate_treat[i] * treat))

            if new_pop_size < 10*self.normalization_factor and self.death_rate_treat[i]*treat > 0:
                new_pop_size = 0
        else:
            raise NotImplementedError

        if flag == 'instant_with_noise' or flag == 'instant_fixed_treat_with_noise' or flag == 'delayed_with_noise':
            # add noise
            # rand = truncnorm(loc=0, scale=0.00528*new_pop_size, a=-0.02/0.00528, b=0.02/0.00528).rvs()
            if new_pop_size < 10 * self.normalization_factor and self.death_rate_treat[i] * self.state[2] > 0:
                new_pop_size = 0
            rand = np.random.normal(0, 0.01 * new_pop_size, 1)[0]
            if np.abs(rand) > 0.05 * new_pop_size:
                rand = 0.05 * new_pop_size * np.sign(rand)
            new_pop_size += rand
        # if new_pop_size < 10 * self.normalization_factor and self.death_rate_treat[i] * self.state[2] > 0:
        #     new_pop_size = 0
        if self.time >= self.end_time:
            new_pop_size = 0
        return new_pop_size


if __name__ == "__main__": # pragma: no cover
    # set random seed
    np.random.seed(int(time.time()))
    env = LvEnv.from_yaml("../../../config.yaml")
    env.reset()

    while not env.done:
        act = env.action_space.sample()
        o, r, t, tr, i = env.step(act)
        print(r)
        print(env.state)

