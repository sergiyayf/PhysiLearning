import numpy as np
from physilearning.envs.base_env import BaseEnv
from physilearning.reward import Reward
from typing import Dict, List, Tuple, Any
import warnings


# Lattice based tumor growth simulation environment for reinforcement learning
# with two populations of cancerous cells, wild type and mutant. Wild type cells
# can be treated with a drug to kill them. Mutant cells are resistant to the drug.


class GridEnv(BaseEnv):
    """
    Lattice based tumor growth simulation environment for reinforcement learning

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
        name: str = 'GridEnv',
        observation_type: str = 'image',
        action_type: str = 'discrete',
        max_tumor_size: int = 600,
        max_time: int = 1000,
        initial_wt: int = 2,
        initial_mut: int = 1,
        growth_rate_wt: float = 0.1,
        growth_rate_mut: float = 0.02,
        death_rate_wt: float = 0.002,
        death_rate_mut: float = 0.002,
        treat_death_rate_wt: float = 0.02,
        treat_death_rate_mut: float = 0.0,
        treatment_time_step: int = 1,
        reward_shaping_flag: int = 0,
        normalize: bool = True,
        normalize_to: float = 1,
        image_size: int = 36,
        patient_id: int = 0,
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

        # GridEnv specific
        if env_specific_params is None:
            env_specific_params = {'cell_positioning': 'random'}
        self.reference_death_rate = self.death_rate
        self.cell_positioning = env_specific_params.get('cell_positioning', 'random')
        self.place_cells(positioning=self.cell_positioning)

    def place_cells(self, positioning: str = 'random') -> None:
        """
        Place cells on the grid
        :param positioning: (str) 'random' or 'surround_mutant'
        """
        if self.normalize:
            ini_wt = int(self.initial_wt/self.normalization_factor)
            ini_mut = int(self.initial_mut/self.normalization_factor)
        else:
            ini_wt = self.initial_wt
            ini_mut = self.initial_mut

        if positioning == 'random':
            # put up to 10 wild type cells in random locations
            for i in range(ini_wt):
                self.image[0, np.random.randint(0, self.image_size),
                           np.random.randint(0, self.image_size)] = self.wt_color

            # put 1 mutant cell in random location
            for j in range(ini_mut):
                pos_x = np.random.randint(0, 10)
                pos_y = np.random.randint(0, 10)
                while self.image[0, pos_x, pos_y] == self.wt_color:
                    pos_x = np.random.randint(0, 10)
                    pos_y = np.random.randint(0, 10)
                self.image[0, pos_x, pos_y] = self.mut_color

        elif positioning == 'surround_mutant':
            pos_x, pos_y = 0, 0
            for i in range(ini_mut):
                pos_x = self.image_size//2
                pos_y = self.image_size//2
                while self.image[0, pos_x, pos_y] != 0:
                    pos_x += np.random.randint(0, 2)
                    pos_y += np.random.randint(0, 2)
                self.image[0, pos_x, pos_y] = self.mut_color
            neighbors = self.check_neighbors(pos_x, pos_y, self.image)
            for i in range(ini_wt):
                rand_neighbor = np.random.randint(0, len(neighbors))
                while self.image[0, neighbors[rand_neighbor][0], neighbors[rand_neighbor][1]] != 0:
                    rand_neighbor = np.random.randint(0, len(neighbors))
                self.image[0, neighbors[rand_neighbor][0], neighbors[rand_neighbor][1]] = self.wt_color

        elif positioning == 'load':
            self.image[0] = np.load('./data/grid_env_data/initial_image.npy')

        else:
            raise ValueError('Positioning method not recognized.')

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment simulating simple tumour growth
        on the grid following the rule that a cell can only grow if it
        has at least one free space around it. The same rule applies to
        cell death by treatment, to first treat the cells that on the edge.

        :param action: (int) the action to
        :return: (np.ndarray, float, bool, Dict[str, Any])
        the next state, the reward, if the episode is done, and additional info
        """
        # grow tumor
        if self.treatment_time_step != 1:
            warnings.warn("Treatment time step is not 1. This is not supported yet for grid env.")
        self.time += self.treatment_time_step
        self.apply_treatment_action(action)
        self.image = self.grow_tumor(self.image)

        num_wt_cells, num_mut_cells = self._get_tumor_volume_from_image(self.image[0, :, :])
        if self.normalize:
            self.state[0] = num_wt_cells * self.normalization_factor
            self.state[1] = num_mut_cells * self.normalization_factor
        else:
            self.state[0] = num_wt_cells
            self.state[1] = num_mut_cells
        self.state[2] = action

        if self.observation_type == 'image' or self.observation_type == 'multiobs':
            self.image_trajectory[:, :, int(self.time / self.treatment_time_step) - 1] = self.image[0, :, :]
            self.done = self._check_done(burden_type='number', total_cell_number=num_wt_cells+num_mut_cells)
            self.trajectory[:, int(self.time / self.treatment_time_step) - 1] = self.state
            rewards = Reward(self.reward_shaping_flag)
            reward = rewards.get_reward(self.state, self.time / self.max_time)
            if self.observation_type == 'image':
                obs = self.image
            elif self.observation_type == 'multiobs':
                obs = {'vec': self.state, 'img': self.image}
            else:
                raise ValueError('Observation type not supported.')

        elif self.observation_type == 'number':
            self.trajectory[:, int(self.time / self.treatment_time_step) - 1] = self.state
            rewards = Reward(self.reward_shaping_flag)
            reward = rewards.get_reward(self.state, self.time / self.max_time)
            obs = self.state
            if self.time >= self.max_time or np.sum(self.state[0:2]) >= self.threshold_burden:
                self.done = True
            else:
                self.done = False
        else:
            raise ValueError('Observation type not supported.')

        # return state, reward, done, info
        return obs, reward, self.done, {}

    def grow_tumor(self, grid: np.ndarray) -> np.ndarray:
        """
        Grow the tumor, check for cancerous check for cancerous cells
        if cancerous cell, check for neighbors with probability of growth rate,
        grow wt cancer cells with probability of growth rate,
        grow mut cancer cells if no neighbors, do nothing

        :param grid: (np.ndarray) the grid
        :return: (np.ndarray) the updated grid
        """
        wt_cells = np.where(grid == self.wt_color)[1:]
        mut_cells = np.where(grid == self.mut_color)[1:]
        # grow wild type cells
        for i in range(len(wt_cells[0])):
            wt_rand = np.random.rand(len(wt_cells[0]))
            # check for neighbors
            neighbors = self.check_neighbors(wt_cells[0][i], wt_cells[1][i], grid)
            # kill cells first
            if self.death_rate[0] == self.reference_death_rate[0]:
                if wt_rand[i] < self.death_rate[0]:
                    grid[0, wt_cells[0][i], wt_cells[1][i]] = 0
            if self.death_rate[0] == self.death_rate_treat[0]:
                if wt_rand[i] < self.reference_death_rate[0]:
                    grid[0, wt_cells[0][i], wt_cells[1][i]] = 0
                if len(neighbors) > 1 and wt_rand[i] < self.death_rate_treat[0]:
                    grid[0, wt_cells[0][i], wt_cells[1][i]] = 0

            # if neighbors and random number is less than growth rate, grow tumor
            if neighbors and wt_rand[i] < self.growth_rate[0]:
                # choose random neighbor
                rand_neighbor = np.random.randint(0, len(neighbors))
                # grow tumor
                grid[0, neighbors[rand_neighbor][0],
                     neighbors[rand_neighbor][1]] = grid[0, wt_cells[0][i], wt_cells[1][i]]

        # grow mutant cells
        for i in range(len(mut_cells[0])):
            # check for neighbors
            mut_rand = np.random.rand(len(mut_cells[0]))
            # kill cells first
            if mut_rand[i] < self.death_rate[1]:
                grid[0, mut_cells[0][i], mut_cells[1][i]] = 0
            neighbors = self.check_neighbors(mut_cells[0][i], mut_cells[1][i], grid)
            # if neighbors and random number is less than growth rate, grow tumor
            if neighbors and mut_rand[i] < self.growth_rate[1]:
                # choose random neighbor
                rand_neighbor = np.random.randint(0, len(neighbors))
                # grow tumor
                grid[0, neighbors[rand_neighbor][0], neighbors[rand_neighbor][1]] =\
                    grid[0, mut_cells[0][i], mut_cells[1][i]]

        return grid

    def _get_tumor_volume_from_image(self, image: np.ndarray) -> tuple:
        """
        Calculate the number of wt and mut cells in the state
        :param image: the state
        :return: number of wt and mut cells
        """
        num_wt_cells = np.sum(image == self.wt_color)
        num_mut_cells = np.sum(image == self.mut_color)
        return num_wt_cells, num_mut_cells

    def apply_treatment_action(self, action: int) -> None:
        """
        Utility function to apply the treatment action
        """
        if action == 0:
            self.death_rate = self.reference_death_rate

        elif action == 1:
            self.death_rate = [self.death_rate_treat[0], self.reference_death_rate[1]]
        return

    def _check_done(self, burden_type: str, **kwargs) -> bool:
        """
        Check if the episode is done: if the tumor is too big or the time is too long
        :param burden_type: type of burden to check
        :return: if the episode is done
        """

        if burden_type == 'number':
            total_cell_number = kwargs['total_cell_number']
        else:
            num_wt_cells = np.sum(kwargs['image'] == self.wt_color)
            num_mut_cells = np.sum(kwargs['image'] == self.mut_color)
            total_cell_number = num_wt_cells + num_mut_cells

        if total_cell_number > self.threshold_burden or self.time >= self.max_time:
            return True
        else:
            return False

    def check_neighbors(
        self,
        x: int,
        y: int,
        grid: np.ndarray,
    ) -> List[List[int]]:
        """
        Check for neighbors. Utility function for growing and treating the tumor
        :param x: (np.uint8) the x coordinate
        :param y: (np.uint8) the y coordinate
        :param grid: (np.ndarray) the grid
        :return: (List[np.ndarray]) the list of neighbors
        """
        # check for neighbors
        neighbors = []
        # check for neighbors
        if x > 0:
            if grid[0, x-1, y] == 0:
                neighbors.append([x-1, y])
        if x < self.image_size-1:
            if grid[0, x+1, y] == 0:
                neighbors.append([x+1, y])
        if y > 0:
            if grid[0, x, y-1] == 0:
                neighbors.append([x, y-1])
        if y < self.image_size-1:
            if grid[0, x, y+1] == 0:
                neighbors.append([x, y+1])
        return neighbors

    def reset(self) -> np.ndarray:
        """
        Reset the environment
        :return: (np.ndarray) the initial state
        """
        # reset time
        self.time = 0
        # reset state
        self.image = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)
        self.place_cells(positioning=self.cell_positioning)
        # put up to 10 wild type cells in random locations

        self.state = [self.initial_wt, self.initial_mut, self.initial_drug]
        # reset done
        self.done = False
        if self.observation_type == 'number':
            obs = self.state
            self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time)))
        elif self.observation_type == 'image' or self.observation_type == 'multiobs':
            self.image_trajectory = np.zeros(
                (self.image_size, self.image_size, int(self.max_time / self.treatment_time_step)))
            self.trajectory = np.zeros(
                (np.shape(self.state)[0], int(self.max_time / self.treatment_time_step)))
            if self.observation_type == 'image':
                obs = self.image
            elif self.observation_type == 'multiobs':
                obs = {'vec': self.state, 'img': self.image}
            else:
                raise ValueError('Observation type not supported')
        else:
            raise ValueError('Observation type not supported')

        return obs


if __name__ == "__main__":
    env = GridEnv.from_yaml("../../../config.yaml")
    env.reset()
    grid = env.image

    while not env.done:
        act = 0  # env.action_space.sample()
        env.step(act)

    anim = env.render()
