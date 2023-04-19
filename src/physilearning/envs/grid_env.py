import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.animation as animation
from matplotlib import pyplot as plt

import yaml
from gym.spaces import Discrete, Box
import numpy as np
from physilearning.envs.base_env import BaseEnv
from stable_baselines3.common.env_checker import check_env
from typing import Dict, List, Tuple, Any


# Lattice based tumor growth simulation environment for reinforcement learning
# with two populations of cancerous cells, wild type and mutant. Wild type cells
# can be treated with a drug to kill them. Mutant cells are resistant to the drug.


class GridEnv(BaseEnv):
    """
    Lattice based tumor growth simulation environment for reinforcement learning

    :param grid_size: (int) Size of the simulation grid
    :param observation_type: (str) Type of observation space.
     Can be 'image' or 'number' or 'multiobs'
    :param action_type: (str) Type of action space. Can be 'discrete' or 'continuous'
    :param normalize: (bool) Whether to normalize the observation space
    :param normalize_to: (float) Value to normalize the observation space to
    :param max_tumor_size: (int) Maximum tumor size
    :param max_time: (int) Maximum time steps
    :param reward_shaping_flag: (int) Flag to use reward shaping
    :param num_wt: (int) Number of wild type cells
    :param num_mut: (int) Number of mutant cells
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
        grid_size: int = 36,
        observation_type: str = 'image',
        action_type: str = 'discrete',
        normalize: bool = True,
        normalize_to: float = 1,
        max_tumor_size: int = 600,
        max_time: int = 1000,
        reward_shaping_flag: int = 0,
        num_wt: int = 2,
        num_mut: int = 1,
        wt_growth_rate: float = 0.1,
        mut_growth_rate: float = 0.02,
        wt_death_rate: float = 0.002,
        mut_death_rate: float = 0.002,
        wt_treat_death_rate: float = 0.02,
        mut_treat_death_rate: float = 0.0,
        cell_positioning = 'surround_mutant'
    ) -> None:
        super().__init__()
        # Configuration
        self.grid_size = grid_size

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
                                         shape=(1,self.grid_size,self.grid_size),
                                         dtype=np.uint8)
        elif self.observation_type == 'number':
            raise NotImplementedError
        elif self.observation_type == 'multiobs':
            raise NotImplementedError

        # Environment parameters
        self.normalize = normalize
        self.normalize_to = normalize_to
        self.threshold_burden = normalize_to
        self.max_tumor_size = max_tumor_size
        self.max_time = max_time
        self.reward_shaping_flag = reward_shaping_flag
        self.done = False

        self.grid = np.zeros((1, self.grid_size, self.grid_size), dtype=np.uint8)
        self.trajectory = np.zeros((self.grid_size, self.grid_size, self.max_time))
        self.num_wt = num_wt
        self.num_mut = num_mut
        self.wt_color = 128
        self.mut_color = 255
        self.wt_growth_rate = wt_growth_rate
        self.mut_growth_rate = mut_growth_rate
        self.reference_wt_death_rate = wt_death_rate
        self.reference_mut_death_rate = mut_death_rate
        self.wt_death_rate = self.reference_wt_death_rate
        self.mut_death_rate = self.reference_mut_death_rate
        self.wt_drug_death_rate = wt_treat_death_rate
        self.mut_drug_death_rate = mut_treat_death_rate
        self.time = 0
        self.cell_positioning = cell_positioning
        self.place_cells(positioning=self.cell_positioning)
        self.fig, self.ax = plt.subplots()

    @classmethod
    def from_yaml(cls, config_file: str = 'config.yaml'):
        """
        Create an environment from a yaml file
        :param config_file: (str) path to the config file
        :return: (object) the environment
        """
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return cls(grid_size=config['env']['GridEnv']['grid_size'],
                   observation_type=config['env']['GridEnv']['observation_type'],
                   action_type=config['env']['GridEnv']['action_type'],
                   normalize=config['env']['normalize'],
                   normalize_to=config['env']['normalize_to'],
                   max_tumor_size=config['env']['threshold_burden'],
                   max_time=config['env']['max_time'],
                   reward_shaping_flag=config['env']['reward_shaping'],
                   num_wt=config['env']['GridEnv']['num_wt'],
                   num_mut=config['env']['GridEnv']['num_mut'],
                   wt_growth_rate=config['env']['GridEnv']['wt_growth_rate'],
                   mut_growth_rate=config['env']['GridEnv']['mut_growth_rate'],
                   wt_death_rate=config['env']['GridEnv']['wt_death_rate'],
                   mut_death_rate=config['env']['GridEnv']['mut_death_rate'],
                   wt_treat_death_rate=config['env']['GridEnv']['wt_treat_death_rate'],
                   mut_treat_death_rate=config['env']['GridEnv']['mut_treat_death_rate'],
                   cell_positioning=config['env']['GridEnv']['cell_positioning']
                   )

    def place_cells(self, positioning: str = 'random') -> None:
        """
        Place cells on the grid
        :param positioning: (str) 'random' or 'surround_mutant'
        """
        if positioning == 'random':
            # put up to 10 wild type cells in random locations
            for i in range(self.num_wt):
                self.grid[0, np.random.randint(0,self.grid_size),
                          np.random.randint(0,self.grid_size)] = self.wt_color

            # put 1 mutant cell in random location
            for j in range(self.num_mut):
                pos_x = np.random.randint(0,10)
                pos_y = np.random.randint(0,10)
                while self.grid[0, pos_x, pos_y] == self.wt_color:
                    pos_x = np.random.randint(0,10)
                    pos_y = np.random.randint(0,10)
                self.grid[0, pos_x, pos_y] = self.mut_color

        elif positioning == 'surround_mutant':
            for i in range(self.num_mut):
                pos_x = self.grid_size//2
                pos_y = self.grid_size//2
                while self.grid[0, pos_x, pos_y] != 0:
                    pos_x += np.random.randint(0, 2)
                    pos_y += np.random.randint(0, 2)
                self.grid[0, pos_x, pos_y] = self.mut_color
            neighbors = self.check_neighbors(pos_x, pos_y, self.grid)
            for i in range(self.num_wt):
                rand_neighbor = np.random.randint(0, len(neighbors))
                while self.grid[0, neighbors[rand_neighbor][0], neighbors[rand_neighbor][1]] != 0:
                    rand_neighbor = np.random.randint(0, len(neighbors))
                self.grid[0, neighbors[rand_neighbor][0], neighbors[rand_neighbor][1]] = self.wt_color


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
        self.apply_treatment_action(action)
        self.grid = self.grow_tumor(self.grid)
        # update state
        self.state = self.grid[0,:,:]
        # update time
        self.time += 1
        # update trajectory
        self.trajectory[:,:,self.time-1] = self.state
        # calculate reward
        #rewards = Reward(self.reward_shaping_flag, normalization=100)
        #reward = rewards.get_reward(self.state, self.time/self.max_time)
        if action: 
            reward = 0
        elif action==0 and np.sum(self.state)<1.e-3: 
            reward = 10
        else:
            reward = 1 
        # check if done
        self.done = self.check_done(self.state)

        # return state, reward, done, info
        return self.grid, reward, self.done, {}

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
                if neighbors and wt_rand[i] < self.wt_drug_death_rate:
                    grid[0, wt_cells[0][i], wt_cells[1][i]] = 0

            # if neighbors and random number is less than growth rate, grow tumor
            if neighbors and wt_rand[i] < self.wt_growth_rate:
                # choose random neighbor
                rand_neighbor = np.random.randint(0,len(neighbors))
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
                rand_neighbor = np.random.randint(0,len(neighbors))
                # grow tumor
                grid[0, neighbors[rand_neighbor][0], neighbors[rand_neighbor][1]] = grid[0, mut_cells[0][i], mut_cells[1][i]]

        return grid

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


    def check_done(self, state: np.ndarray) -> bool:
        """
        Check if the episode is done: if the tumor is too big or the time is too long
        :param state: (np.ndarray) the state
        :return: (bool) if the episode is done
        """
        # check if done
        num_wt_cells = np.sum(state == self.wt_color)
        num_mut_cells = np.sum(state == self.mut_color)
        total_cell_number = num_wt_cells + num_mut_cells
        if total_cell_number > self.max_tumor_size or self.time >= self.max_time:
            return True
        else:
            return False

    def check_neighbors(
        self,
        x: np.uint8 ,
        y: np.uint8, grid: np.ndarray
    ) -> List[np.ndarray]:
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
            if grid[0, x-1,y] == 0:
                neighbors.append([x-1,y])
        if x < self.grid_size-1:
            if grid[0, x+1,y] == 0:
                neighbors.append([x+1,y])
        if y > 0:
            if grid[0, x,y-1] == 0:
                neighbors.append([x,y-1])
        if y < self.grid_size-1:
            if grid[0, x,y+1] == 0:
                neighbors.append([x,y+1])
        return neighbors


    def reset(self) -> np.ndarray:
        """
        Reset the environment
        :return: (np.ndarray) the initial state
        """
        # reset time
        self.time = 0
        # reset state
        self.grid = np.zeros((1, self.grid_size, self.grid_size), dtype=np.uint8)
        self.place_cells(positioning=self.cell_positioning)
        # put up to 10 wild type cells in random locations

        self.state = self.grid
        # reset trajectory
        self.trajectory = np.zeros((self.grid_size, self.grid_size, self.max_time))
        # reset done
        self.done = False
        # reset reward
        self.reward = 0
        # return state
        return self.grid

    def render(self, mode: str = 'human') -> mpl.animation.ArtistAnimation:
        # render state
        # plot it on the grid with different colors for wt and mut
        # animate simulation with matplotlib animation
        ims = []

        for i in range(self.time):
            im = self.ax.imshow(self.trajectory[:,:,i], animated=True, cmap='viridis', vmin=0, vmax=255)
            ims.append([im])
        ani = animation.ArtistAnimation(self.fig, ims, interval=0.1, blit=True, repeat_delay=1000)

        return ani

    def close(self):
        pass


if __name__ == "__main__":
    env = GridEnv.from_yaml("../../../config.yaml")
    env.reset()
    grid = env.grid

    while not env.done:
        action = 0 #env.action_space.sample()
        env.step(action)

    anim = env.render()
    plt.show()
    check_env(env, warn=True)

