import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import yaml
from gym.spaces import Discrete, Box
import numpy as np
from physilearning.envs.base_env import BaseEnv
from physilearning.reward import Reward
from stable_baselines3.common.env_checker import check_env
from typing import Dict, List, Tuple, Any
import warnings


# Lattice based tumor growth simulation environment for reinforcement learning
# with two populations of cancerous cells, wild type and mutant. Wild type cells
# can be treated with a drug to kill them. Mutant cells are resistant to the drug.


class GridEnv(BaseEnv):
    """
    Lattice based tumor growth simulation environment for reinforcement learning

    :param image_size: (int) Size of the simulation grid
    :param observation_type: (str) Type of observation space.
     Can be 'image' or 'number' or 'multiobs'
    :param action_type: (str) Type of action space. Can be 'discrete' or 'continuous'
    :param normalize: (bool) Whether to normalize the observation space
    :param normalize_to: (float) Value to normalize the observation space to
    :param max_tumor_size: (int) Maximum tumor size
    :param max_time: (int) Maximum time steps
    :param reward_shaping_flag: (int) Flag to use reward shaping
    :param initial_wt: (int) Number of wild type cells
    :param initial_mut: (int) Number of mutant cells
    :param wt_growth_rate: (float) Growth rate of wild type cells
    :param mut_growth_rate: (float) Growth rate of mutant cells
    :param wt_death_rate: (float) Death rate of wild type cells
    :param mut_death_rate: (float) Death rate of mutant cells
    :param wt_treat_death_rate: (float) Death rate of wild type cells when treated
    :param mut_treat_death_rate: (float) Death rate of mutant cells when treated
    :param cell_positioning: (str) Method to position cells.
     Can be 'random' or 'surround_mut'
    """

    def __init__(
        self,
        observation_type: str = 'image',
        action_type: str = 'discrete',
        image_size: int = 36,
        normalize: bool = True,
        normalize_to: float = 1,
        max_tumor_size: int = 600,
        max_time: int = 1000,
        treatment_time_step: int = 1,
        reward_shaping_flag: int = 0,
        initial_wt: int = 2,
        initial_mut: int = 1,
        wt_growth_rate: float = 0.1,
        mut_growth_rate: float = 0.02,
        wt_death_rate: float = 0.002,
        mut_death_rate: float = 0.002,
        wt_treat_death_rate: float = 0.02,
        mut_treat_death_rate: float = 0.0,
        cell_positioning: str = 'surround_mutant',
    ) -> None:
        super().__init__()
        #################### Todo: move to base class ##################
        # Spaces
        self.name = 'GridEnv'
        self.action_type = action_type
        if self.action_type == 'discrete':
            self.action_space = Discrete(2)
        elif self.action_type == 'continuous':
            self.action = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_type = observation_type
        if self.observation_type == 'image':
            self.observation_space = Box(low=0, high=255,
                                         shape=(1, image_size, image_size),
                                         dtype=np.uint8)
        elif self.observation_type == 'number':
            self.observation_space = Box(low=0, high=1, shape=(1,))
        elif self.observation_type == 'multiobs':
            raise NotImplementedError
        else:
            raise NotImplementedError

        # Configurations
        self.image_size = image_size
        self.normalize = normalize
        self.max_tumor_size = max_tumor_size
        self.normalization_factor = normalize_to/max_tumor_size
        self.reward_shaping_flag = reward_shaping_flag
        self.image = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)
        self.time = 0
        self.max_time = max_time
        self.treatment_time_step = treatment_time_step
        self.wt_color = 128
        self.mut_color = 255
        self.drug_color = 0
        self.initial_drug = 0
        self.done = False

        # set up initial conditions
        if self.normalize:
            self.threshold_burden = normalize_to
            self.initial_wt = initial_wt*self.normalization_factor
            self.initial_mut = initial_mut*self.normalization_factor
        else:
            self.threshold_burden = max_tumor_size
            self.initial_wt = initial_wt
            self.initial_mut = initial_mut

        # set up initial state
        self.state = [self.initial_wt,
                      self.initial_mut,
                      self.initial_drug]

        # trajectory for plotting
        if self.observation_type == 'number':
            self.trajectory = np.zeros((self.max_time, 1))
        elif self.observation_type == 'image':
            self.trajectory = np.zeros((self.image_size, self.image_size, int(self.max_time/self.treatment_time_step)))
            self.number_trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time/self.treatment_time_step)))

        ###############################################
        # GridEnv specific for now
        self.wt_growth_rate = wt_growth_rate
        self.mut_growth_rate = mut_growth_rate
        self.reference_wt_death_rate = wt_death_rate
        self.reference_mut_death_rate = mut_death_rate
        self.wt_death_rate = self.reference_wt_death_rate
        self.mut_death_rate = self.reference_mut_death_rate
        self.wt_drug_death_rate = wt_treat_death_rate
        self.mut_drug_death_rate = mut_treat_death_rate

        self.cell_positioning = cell_positioning
        self.place_cells(positioning=self.cell_positioning)
        self.fig, self.ax = plt.subplots()

    @classmethod
    def from_yaml(cls, config_file: str = 'config.yaml', port: str = '0', job_name: str = '000000'):
        """
        Create an environment from a yaml file
        :param config_file: (str) path to the config file
        :param port: (str) port to use for the environment
        :param job_name: (str) job name
        :return: (object) the environment

        """
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return cls(image_size=config['env']['image_size'],
                   observation_type=config['env']['observation_type'],
                   action_type=config['env']['action_type'],
                   normalize=config['env']['normalize'],
                   normalize_to=config['env']['normalize_to'],
                   max_tumor_size=config['env']['max_tumor_size'],
                   max_time=config['env']['max_time'],
                   reward_shaping_flag=config['env']['reward_shaping'],
                   initial_wt=config['env']['GridEnv']['initial_wt'],
                   initial_mut=config['env']['GridEnv']['initial_mut'],
                   wt_growth_rate=config['env']['GridEnv']['wt_growth_rate'],
                   mut_growth_rate=config['env']['GridEnv']['mut_growth_rate'],
                   wt_death_rate=config['env']['GridEnv']['wt_death_rate'],
                   mut_death_rate=config['env']['GridEnv']['mut_death_rate'],
                   wt_treat_death_rate=config['env']['GridEnv']['wt_treat_death_rate'],
                   mut_treat_death_rate=config['env']['GridEnv']['mut_treat_death_rate'],
                   cell_positioning=config['env']['GridEnv']['cell_positioning'],
                   treatment_time_step=config['env']['treatment_time_step'],
                   )

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

        if self.observation_type == 'image':
            self.trajectory[:, :, int(self.time / self.treatment_time_step) - 1] = self.image[0, :, :]
            obs = self.image
            self.done = self._check_done(burden_type='number', total_cell_number=num_wt_cells+num_mut_cells)
            self.number_trajectory[:, int(self.time / self.treatment_time_step) - 1] = self.state
            rewards = Reward(self.reward_shaping_flag)
            reward = rewards.get_reward(self.state, self.time / self.max_time)

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
            if self.wt_death_rate == self.reference_wt_death_rate:
                if wt_rand[i] < self.wt_death_rate:
                    grid[0, wt_cells[0][i], wt_cells[1][i]] = 0
            if self.wt_death_rate == self.wt_drug_death_rate:
                if wt_rand[i] < self.reference_wt_death_rate:
                    grid[0, wt_cells[0][i], wt_cells[1][i]] = 0
                if len(neighbors) > 1 and wt_rand[i] < self.wt_drug_death_rate:
                    grid[0, wt_cells[0][i], wt_cells[1][i]] = 0

            # if neighbors and random number is less than growth rate, grow tumor
            if neighbors and wt_rand[i] < self.wt_growth_rate:
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
            if mut_rand[i] < self.mut_death_rate:
                grid[0, mut_cells[0][i], mut_cells[1][i]] = 0
            neighbors = self.check_neighbors(mut_cells[0][i], mut_cells[1][i], grid)
            # if neighbors and random number is less than growth rate, grow tumor
            if neighbors and mut_rand[i] < self.mut_growth_rate:
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
            self.wt_death_rate = self.reference_wt_death_rate
            self.mut_death_rate = self.reference_mut_death_rate

        elif action == 1:
            self.wt_death_rate = self.wt_drug_death_rate
            self.mut_death_rate = self.reference_mut_death_rate
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
            obs = [np.sum(self.state[0:2])]
            self.trajectory = np.zeros((self.image_size, self.image_size, self.max_time))
        elif self.observation_type == 'image':
            obs = self.image
            self.trajectory = np.zeros(
                (self.image_size, self.image_size, int(self.max_time / self.treatment_time_step)))
            self.number_trajectory = np.zeros(
                (np.shape(self.state)[0], int(self.max_time / self.treatment_time_step)))
        else:
            raise ValueError('Observation type not supported')

        return obs

    def render(self, mode: str = 'human') -> mpl.animation.ArtistAnimation:
        # render state
        # plot it on the grid with different colors for wt and mut
        # animate simulation with matplotlib animation
        ims = []

        for i in range(self.time):
            im = self.ax.imshow(self.trajectory[:, :, i], animated=True, cmap='viridis', vmin=0, vmax=255)
            ims.append([im])
        ani = animation.ArtistAnimation(self.fig, ims, interval=0.1, blit=True, repeat_delay=1000)

        return ani

    def close(self):
        pass


if __name__ == "__main__":
    env = GridEnv.from_yaml("../../../config.yaml")
    env.reset()
    grid = env.image

    while not env.done:
        act = 0  # env.action_space.sample()
        env.step(act)

    anim = env.render()
    plt.show()
    check_env(env, warn=True)
