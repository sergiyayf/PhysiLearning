import numpy as np
from physilearning.envs.base_env import BaseEnv
from physilearning.reward import Reward
from typing import Tuple
import time
from physilearning.envs.base_env import trunc_norm


class KppEnv(BaseEnv):
    """
    Environment for density dependent tumor growth model

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
        name: str = 'KppEnv',
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
        see_resistance: bool = False,
        see_prev_action: bool = False,
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
                         see_resistance=see_resistance, see_prev_action=see_prev_action,
                         env_specific_params=env_specific_params,
                         )
        self.env_specific_params = env_specific_params
        #self.time_end = self.params['timestep_size'] * (self.params['num_timesteps'] - 1)
        #self.time_array = np.arange(self.params['num_timesteps']) * self.params['timestep_size']

        self.radial_step_size = (self.env_specific_params['r_max']-self.env_specific_params['r_min'])/(self.env_specific_params['r_bins']-1)
        self.radius_array = np.linspace(self.env_specific_params['r_min'], self.env_specific_params['r_max'], self.env_specific_params['r_bins'])
        self.sensitive_population, self.resistant_population = None, None
        self.growth_fraction = 1
        self.death_fraction = 0
        self.current_sensitive_growth_rate = self.growth_rate[0]*self.growth_fraction
        self.current_treat_death_rate = self.death_rate_treat[0]*self.death_fraction
        self.min_death_during_treat = self.env_specific_params['min_death_during_treat']
        self.timestep_size = self.env_specific_params['timestep_size']
        self.sensitive_population, self.resistant_population = None, None
        self.initialize_population()

        # rewrite initial state and normalizations
        self.initial_wt = self.density_to_number(self.sensitive_population)
        self.initial_mut = self.density_to_number(self.resistant_population)
        if self.normalize:
            self.normalization_factor = self.normalize_to/(self.initial_wt + self.initial_mut)
            self.initial_wt *= self.normalization_factor
            self.initial_mut *= self.normalization_factor
            self.threshold_burden = self.max_tumor_size * self.normalize_to
            self.state = [self.initial_wt, self.initial_mut, 0]
            self.trajectory[:,0] = self.state

        self.density_trajectory = np.zeros((self.env_specific_params['r_bins'], self.max_time+1, 2))
        self.density_trajectory[:,0,0] = self.sensitive_population
        self.density_trajectory[:,0,1] = self.resistant_population

    def initialize_population(self):
        prms = {}
        for key in ['sensitive_colony_radius', 'peak_center', 'sigma']:
            if isinstance(self.env_specific_params[key], str):
                value = self.env_specific_params[key]
                if '-' in value:
                    low, high = (float(val) for val in value.split('-'))
                    val = np.random.uniform(low, high)
                elif 'pm' in value:
                    mean = float(value.split('pm')[0])
                    std = float(value.split('pm')[1])
                    val = trunc_norm(mean, std, 3)
                else:
                    raise ValueError(f"Invalid value for {key}")
            else:
                val = self.env_specific_params[key]
            prms[key] = val
        # account for correct radial distribution of resistant cells
        prms['peak_center'] = np.sqrt((prms['sensitive_colony_radius']-1)*prms['peak_center'])

        sensitive_initialization = self.initialization_sigmoid(prms['sensitive_colony_radius'])
        resistant_initialization = 1.0 * self.initialization_gaussian(prms['peak_center'], prms['sigma'])
        sensitive_initialization -= resistant_initialization
        self.sensitive_population = sensitive_initialization
        self.resistant_population = resistant_initialization
        return

    def density_to_number(self, density):
        num = 2*np.pi*np.dot(density,self.radius_array)
        # add noise
        num += np.random.normal(0, 0.05*num)
        if num < 0:
            num = 0
        return num

    @staticmethod
    def initialization_sigmoid(colony_radius):
        x = np.linspace(0, 699, 700)
        return 1 / (1 + np.exp(0.05 * (x - colony_radius)))
        #return 1 / (1 + np.exp(0.06 * (x - colony_radius)))

    @staticmethod
    def initialization_gaussian(peak_center, sigma):
        x = np.linspace(0, 699, 700)
        return .01 * np.exp(-((x - peak_center) ** 2) / (2 * sigma ** 2))

    def calculate_no_flux_laplacian(self, radial_population_array, density_dep_diffusion_coefficient):
        """
        Compute second derivative under no-flux BC in 1D by mirroring boundary values.
        """
        radial_population_array_length = len(radial_population_array)

        # Extended array for boundary conditions
        ext_radial_population_array = np.zeros(radial_population_array_length + 2)
        ext_radial_population_array[1:-1] = radial_population_array[:]
        ext_density_dep_diffusion_coefficient = np.zeros(radial_population_array_length + 2)
        ext_density_dep_diffusion_coefficient[1:-1] = density_dep_diffusion_coefficient[:]

        # Mirror boundaries
        ext_radial_population_array[0] = ext_radial_population_array[1]
        ext_radial_population_array[-1] = ext_radial_population_array[-2]
        ext_density_dep_diffusion_coefficient[0] = ext_density_dep_diffusion_coefficient[1]
        ext_density_dep_diffusion_coefficient[-1] = ext_density_dep_diffusion_coefficient[-2]

        # Central difference
        laplacian = ((ext_density_dep_diffusion_coefficient[2:] * (
                    ext_radial_population_array[2:] - ext_radial_population_array[1:-1]) -
                      ext_density_dep_diffusion_coefficient[1:-1] * (
                                  ext_radial_population_array[1:-1] - ext_radial_population_array[:-2])) /
                     (self.radial_step_size ** 2))

        return laplacian

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
            s,r = self.grow()
            self.state[0] = s*self.normalization_factor
            self.state[1] = r*self.normalization_factor
            self.burden = np.sum(self.state[0:2])

            self.trajectory[:, self.time] = self.state
            self.density_trajectory[:, self.time, 0] = self.sensitive_population
            self.density_trajectory[:, self.time, 1] = self.resistant_population
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
            self.image_trajectory[:, :, int(self.time / self.treatment_time_step)] = self.image[0, :, :]
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
                obs = [self.state, self.mutant_normalized_position * self.normalization_factor]
            else:
                obs = [np.sum(self.state[0:2]), self.state[2],
                       self.mutant_normalized_position * self.normalization_factor]
        else:
            raise NotImplementedError
        terminate = self.terminate()
        truncate = self.truncate()
        self.done = terminate or truncate

        return obs, reward, terminate, truncate, info

    def reset(self, *, seed=None, options=None):
        self.growth_fraction = 1
        self.death_fraction = 0
        self.current_sensitive_growth_rate = self.growth_rate[0] * self.growth_fraction
        self.current_treat_death_rate = self.death_rate_treat[0] * self.death_fraction
        self.initialize_population()
        self.initial_wt = np.sum(self.sensitive_population * 2 * np.pi * self.radius_array)
        self.initial_mut = np.sum(self.resistant_population * 2 * np.pi * self.radius_array)
        if self.normalize:
            self.normalization_factor = self.normalize_to / (self.initial_wt + self.initial_mut)
            self.initial_wt *= self.normalization_factor
            self.initial_mut *= self.normalization_factor
            self.threshold_burden = self.max_tumor_size * self.normalize_to
        self.state = [self.initial_wt, self.initial_mut, 0]
        self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time) + 1))
        self.trajectory[:, 0] = self.state
        self.time = 0
        self.done = False

        self.density_trajectory = np.zeros((self.env_specific_params['r_bins'], self.max_time+1, 2))
        self.density_trajectory[:, 0, 0] = self.sensitive_population
        self.density_trajectory[:, 0, 1] = self.resistant_population

        self.burden = np.sum(self.state[0:2])

        for t in range(0, int(self.treatment_time_step/2)):
            # step time
            self.time += 1
            self.state[2] = 0
            s,r = self.grow()
            self.state[0] = s*self.normalization_factor
            self.state[1] = r*self.normalization_factor
            self.burden = np.sum(self.state[0:2])

            self.trajectory[:, self.time] = self.state
            self.density_trajectory[:, self.time, 0] = self.sensitive_population
            self.density_trajectory[:, self.time, 1] = self.resistant_population
            # check if done
            if self.state[0] <= 0 and self.state[1] <= 0:
                self.state = [0, 0, 0]

        self.reward = 0
        if self.observation_type == 'number':
            if self.see_resistance:
                obs = self.state[0:2]
            else:
                obs = [np.sum(self.state[0:2])]
            if self.see_prev_action:
                obs = np.append(obs, 0)


        return obs, {}

    def grow(self):

        # 1. update growth and death rate
        step = self.timestep_size / self.env_specific_params['ramp_time']

        if self.state[2]:
            if self.growth_fraction > 0:
                self.growth_fraction -= step
                if self.growth_fraction < 0:
                    self.growth_fraction = 0
            else:
                self.death_fraction += step
                if self.death_fraction > 1:
                    self.death_fraction = 1
        else:
            if self.death_fraction > 0:
                self.death_fraction -= step
                if self.death_fraction < 0:
                    self.death_fraction = 0
            else:
                self.growth_fraction += step
                if self.growth_fraction > 1:
                    self.growth_fraction = 1

        # 2- update step
        self.current_sensitive_growth_rate = self.growth_rate[0] * self.growth_fraction
        self.current_treat_death_rate = self.death_rate_treat[0] * self.death_fraction

        total_density = self.sensitive_population + self.resistant_population

        density_dep_diffusion_coef_sen = self.env_specific_params['diffusion_coefficient'] * (1 - total_density) ** self.env_specific_params[
            'density_exponent_sensitive']
        density_dep_diffusion_coef_res = self.env_specific_params['diffusion_coefficient'] * (1 - total_density) ** self.env_specific_params[
            'density_exponent_resistant']
        density_dep_diffusion_coef_sen[density_dep_diffusion_coef_sen < 0] = 0
        density_dep_diffusion_coef_res[density_dep_diffusion_coef_res < 0] = 0

        laplacian_sensitive = self.calculate_no_flux_laplacian(self.sensitive_population,
                                                               density_dep_diffusion_coef_sen)
        laplacian_resistant = self.calculate_no_flux_laplacian(self.resistant_population,
                                                               density_dep_diffusion_coef_res)

        growth_sensitive = self.current_sensitive_growth_rate * self.sensitive_population * (1 - total_density)
        growth_resistant = self.growth_rate[1] * self.resistant_population * (1 - total_density)

        treat_death_sensitive = self.current_treat_death_rate * self.sensitive_population
        if self.death_fraction > 0:
            treat_death_sensitive += self.min_death_during_treat
        random_death_sensitive = self.death_rate[0] * self.sensitive_population
        random_death_resistant = self.death_rate[1] * self.resistant_population

        self.sensitive_population = (self.sensitive_population + self.timestep_size *
                                     (laplacian_sensitive + growth_sensitive - treat_death_sensitive - random_death_sensitive))
        self.resistant_population = (self.resistant_population + self.timestep_size *
                                     (laplacian_resistant + growth_resistant - random_death_resistant))

        self.sensitive_population[self.sensitive_population < 1.e-4] = 0
        self.resistant_population[self.resistant_population < 1.e-4] = 0

        pop_sens = self.density_to_number(self.sensitive_population)
        pop_res = self.density_to_number(self.resistant_population)

        return pop_sens, pop_res

if __name__ == "__main__": # pragma: no cover
    # set random seed
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    np.random.seed(int(time.time()))
    env = KppEnv.from_yaml("../../../config.yaml")
    env.reset()
    treat = []
    wt = []
    mut = []
    treat.append(env.state[2])
    wt.append(env.state[0])
    mut.append(env.state[1])

    while not env.done:
        act = 1 #env.action_space.sample()
        o, r, t, tr, i = env.step(act)
        #print(r)
        #print(env.state)
        treat.append(env.state[2])
        wt.append(env.state[0])
        mut.append(env.state[1])

    fig, ax = plt.subplots(1, 1)
    tot = np.array(wt) + np.array(mut)
    # normalize to 1
    wt = np.array(wt)#/tot[0]
    mut = np.array(mut)#/tot[0]
    time = 2*np.arange(len(treat))
    ax.plot(time,wt, label='wt')
    ax.plot(time,mut, label='mut')
    ax.fill_between(time, 1, 1.1, where=treat, color='orange', alpha=0.3,
                    label='treatment')
    ax.set_xlabel('time')
    ax.set_ylabel('number')
    ax.set_yscale('linear')
    ax.legend()
    ax.set_xlim(0, 50)
    ax.set_yscale('linear')

    # plot initial density profile
    fig, ax = plt.subplots(1, 1)
    sens_density = env.density_trajectory[:,0,0]
    res_density = env.density_trajectory[:,0,1]
    ax.plot(env.radius_array, sens_density, label='initial')
    ax.plot(env.radius_array, res_density, label='initial', c='r')

    sens_density = env.density_trajectory[:, 10000, 0]
    ax.plot(env.radius_array, sens_density, label='final')
    ax.legend()

