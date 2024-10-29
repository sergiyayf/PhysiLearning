import numpy as np
from physilearning.envs.base_env import BaseEnv
from physilearning.reward import Reward
from typing import Tuple
import time
from scipy.stats import truncnorm


class ArrEnv(BaseEnv):
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
        name: str = 'ArrEnv',
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

        self.trajectory[:, 0] = self.state

        self.mtd_rew = 0
        self.current_rew = 0

        # arrest relevant parameters
        self.sensitive = self.initial_wt
        self.arrested = 0
        self.resistant = self.initial_mut

        self.arrest_rate = env_specific_params.get('arrest_rate', 0.01)
        self.recover_rate = env_specific_params.get('recover_rate', 0.01)

        self.time_on_treatment = 0
        self.time_off_treatment = 0
        self.arresting_time_threshold = env_specific_params.get('arresting_time_threshold', 2)
        self.recovering_time_threshold = env_specific_params.get('recovering_time_threshold', 4)
        self.dying_time_threshold = env_specific_params.get('dying_time_threshold', 2)
        self.dying_time_threshold += self.arresting_time_threshold
        self.cycle_number = 0
        self.initial_growth_reduction = env_specific_params.get('initial_growth_reduction', 0.5)

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
            sus, arr, res = self.grow()
            self.sensitive = sus
            self.arrested = arr
            self.resistant = res
            self.state[0] = sus+arr
            self.state[1] = res
            self.burden = np.sum(self.state[0:2])

            if action == 1:
                self.time_on_treatment += 1
                self.time_off_treatment = 0
            else:
                if self.time_on_treatment > 1:
                    self.cycle_number += 1
                self.time_on_treatment = 0
                self.time_off_treatment += 1
            # record trajectory
            #self.state[2] = action
            #self.trajectory[:, self.time] = self.state

            # check if done
            # if self.state[0] <= 0 and self.state[1] <= 0:
            #     self.state = [0, 0, 0]

            # get the reward
            rewards = Reward(self.reward_shaping_flag, normalization=np.sum(self.trajectory[0:2, 0]))
            if self.reward_shaping_flag == 'tendayaverage':
                reward += rewards.tendayaverage(self.trajectory, self.time)
            elif self.reward_shaping_flag == 'mtd_compare':
                reward += rewards.tendayaverage(self.trajectory, self.time)
            else:
                reward += rewards.get_reward(self.state, self.time/self.max_time, self.threshold_burden)

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

        self.randomize_params()
        if self.normalize:
            keys = self.random_params.keys()
            if 'initial_wt' in keys or 'initial_mut' in keys:
                self.normalization_factor = self.normalize_to / (self.initial_wt + self.initial_mut)
                self.initial_wt *= self.normalization_factor
                self.initial_mut *= self.normalization_factor
                self.capacity = self.capacity_non_normalized * self.normalization_factor

        self.state = [self.initial_wt, self.initial_mut, 0]
        self.sensitive = self.initial_wt
        self.arrested = 0
        self.resistant = self.initial_mut
        self.time = 0
        self.current_rew = 0
        self.done = False
        self.time_on_treatment = 0
        self.time_off_treatment = 0

        self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time)+1))
        self.trajectory[:, 0] = self.state

        if self.observation_type == 'number':
            if self.see_resistance:
                obs = self.state[0:2]
            else:
                obs = [np.sum(self.state[0:2])]

        # do day zero without treatment
        # for tt in [0, 1]:
        #     self.time += 1
        #     sus, arr, res = self.grow()
        #     self.sensitive = sus
        #     self.arrested = arr
        #     self.resistant = res
        #     self.state[0] = sus + arr
        #     self.state[1] = res
        #     self.time_off_treatment += 1
        #     self.burden = np.sum(self.state[0:2])
            # record trajectory
            # self.state[2] = action
            #self.trajectory[:, self.time] = self.state
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
                sus, arr, res = self.grow()
                self.sensitive = sus
                self.arrested = arr
                self.resistant = res
                self.state[0] = sus + arr
                self.state[1] = res
                self.time_on_treatment += 1
                #self.trajectory[:, self.time] = self.state
                rew += Reward(self.reward_shaping_flag, normalization=np.sum(self.trajectory[0:2, 0])).tendayaverage(self.trajectory, self.time)
            terminate = self.terminate()
            truncate = self.truncate()
            self.done = terminate or truncate
        self.done = False
        if rew == 0:
            rew = 100
        return rew

    def grow(self) -> float:
        drug_seen_factor = 1
        recovery_factor = 1
        if self.cycle_number >= 1:
            drug_seen_factor = 2
        if self.cycle_number > 1:
            recovery_factor = 2
        if self.time_on_treatment >= drug_seen_factor * self.arresting_time_threshold:
            arresting = 1
        else:
            arresting = 0

        if self.time_off_treatment >= self.recovering_time_threshold / recovery_factor:
            recovering = 1
        else:
            recovering = 0

        if (self.time_on_treatment >= self.dying_time_threshold and self.state[2] == 1) or (
                self.time_off_treatment < self.recovering_time_threshold + 2 and self.state[2] == 0):
            dying = 1
        else:
            dying = 0

        # Initial phase growth reduction ?? might be essential to be able to fit the data

        sus_gr_rate = self.growth_rate[0]
        res_gr_rate = self.growth_rate[1]
        competition = (1 - (self.sensitive + self.arrested + self.resistant) / self.capacity)
        new_sensitive = self.sensitive * (
                    1 + sus_gr_rate * competition - self.arrest_rate * arresting - self.death_rate[0] * sus_gr_rate) + \
                        self.recover_rate * self.arrested * recovering
        new_arrested = self.arrested * (1 - self.recover_rate * recovering - self.death_rate[0] * sus_gr_rate -
                                        self.death_rate_treat[
                                            0] * dying) + self.sensitive * self.arrest_rate * arresting
        new_resistant = self.resistant * (1 + res_gr_rate * competition - self.death_rate[1] * res_gr_rate)

        # add noise comment for fitting
        # if new_sensitive <= 0:
        #     new_sensitive = 0
        # else:
        #     new_sensitive = np.max([0, new_sensitive + np.random.normal(0, 0.01*new_sensitive)])
        # if new_arrested <= 0:
        #     new_arrested = 0
        # else:
        #     new_arrested = np.max([0, new_arrested + np.random.normal(0, 0.01*new_arrested)])
        # if new_resistant <= 0:
        #     new_resistant = 0
        # else:
        #     new_resistant = np.max([0, new_resistant + np.random.normal(0, 0.01*new_resistant)])


        return new_sensitive, new_arrested, new_resistant


if __name__ == "__main__": # pragma: no cover
    # set random seed
    np.random.seed(int(time.time()))
    env = ArrEnv.from_yaml("../../../config.yaml")
    env.reset()

    while not env.done:
        act = env.action_space.sample()
        o, r, t, tr, i = env.step(act)
        print(r)
        print(env.state)

