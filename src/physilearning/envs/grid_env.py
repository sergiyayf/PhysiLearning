# imports
import yaml
from gym import Env
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from gym.spaces import Discrete, Box
import numpy as np
from physilearning.envs.base_env import BaseEnv
from physilearning.reward import Reward
import matplotlib.animation as animation
from stable_baselines3.common.env_checker import check_env

# Lattice based tumor growth simulation environment for reinforcement learning
# with two populations of cancerous cells, wild type and mutant. Wild type cells
# can be treated with a drug to kill them. Mutant cells are resistant to the drug.

class GridEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        # Configuration
        self.grid_size = 36

        # Spaces
        self.type = 'GridEnv'
        self.action_type = 'discrete'
        self.action_space = Discrete(2)
        self.observation_type = 'box'
        self.observation_space = Box(low=0, high=255, shape=(self.grid_size,self.grid_size,1), dtype=np.uint8)

        # Environment parameters
        self.normalize = False
        self.normalize_to = 1000
        self.max_tumor_size = self.grid_size**2-50
        self.max_time = 1000
        self.reward_shaping_flag = 0

        self.grid = np.zeros((self.grid_size,self.grid_size,1), dtype=np.uint8)
        self.trajectory = np.zeros((self.grid_size,self.grid_size,self.max_time))
        self.num_wt = 2
        self.num_mut = 1
        self.wt_growth_rate = 0.1
        self.mut_growth_rate = 0.01
        self.reference_wt_death_rate = 0.005
        self.reference_mut_death_rate = 0.005
        self.wt_death_rate = self.reference_wt_death_rate
        self.mut_death_rate = self.reference_mut_death_rate
        self.wt_drug_death_rate = 0.1
        self.mut_drug_death_rate = 0.005
        self.time = 0
        self.positioning = 'surround'
        self.place_cells(positioning=self.positioning)
        self.fig, self.ax = plt.subplots()

    def place_cells(self, positioning='random'):
        if positioning == 'random':
            # put up to 10 wild type cells in random locations
            for i in range(self.num_wt):
                self.grid[np.random.randint(0,self.grid_size), np.random.randint(0,self.grid_size)] = 1

            # put 1 mutant cell in random location, check if it is on top of a wild type cell
            for j in range(self.num_mut):
                pos_x = np.random.randint(0,10)
                pos_y = np.random.randint(0,10)
                while self.grid[pos_x, pos_y] == 1:
                    pos_x = np.random.randint(0,10)
                    pos_y = np.random.randint(0,10)
                self.grid[pos_x, pos_y] = 2

        elif positioning == 'surround':
            for i in range(self.num_mut):
                pos_x = np.random.randint(0, 10)
                pos_y = np.random.randint(0, 10)
                while self.grid[pos_x, pos_y] != 0:
                    pos_x = np.random.randint(0, 10)
                    pos_y = np.random.randint(0, 10)
                self.grid[pos_x, pos_y] = 2
            neighbors = self.check_neighbors(pos_x, pos_y, self.grid)
            for i in range(self.num_wt):
                rand_neighbor = np.random.randint(0, len(neighbors))
                self.grid[neighbors[rand_neighbor][0], neighbors[rand_neighbor][1]] = 1


    def step(self, action):
        # grow tumor
        self.apply_treatment_action(action)
        self.grid = self.grow_tumor(self.grid)
        # update state
        self.state = self.grid[:,:,0]
        # update time
        self.time += 1
        # update trajectory
        self.trajectory[:,:,self.time-1] = self.state
        # calculate reward
        rewards = Reward(self.reward_shaping_flag, normalization=100)
        reward = rewards.get_reward(self.state, self.time/self.max_time)
        # check if done
        self.done = self.check_done(self.state)
        # return state, reward, done, info
        return self.grid, reward, self.done, {}

    def grow_tumor(self, grid):
        # grow tumor
        # check for cancerous cells
        # if cancerous cell, check for neighbors
        # with probability of growth rate, grow wt cancer cells
        # with probability of growth rate, grow mut cancer cells
        # if no neighbors, do nothing

        wt_cells = np.where(grid == 1)
        mut_cells = np.where(grid == 2)

        # grow wild type cells
        for i in range(len(wt_cells[0])):
            wt_rand = np.random.rand(len(wt_cells[0]))
            # check for neighbors
            neighbors = self.check_neighbors(wt_cells[0][i], wt_cells[1][i], grid)
            # kill cells first
            if self.wt_death_rate == self.reference_wt_death_rate:
                if wt_rand[i] < self.wt_death_rate:
                    grid[wt_cells[0][i], wt_cells[1][i]] = 0
            if self.wt_death_rate == self.wt_drug_death_rate:
                if wt_rand[i] < self.reference_wt_death_rate:
                    grid[wt_cells[0][i], wt_cells[1][i]] = 0
                if neighbors and wt_rand[i] < self.wt_drug_death_rate:
                    grid[wt_cells[0][i], wt_cells[1][i]] = 0

            # if neighbors and random number is less than growth rate, grow tumor
            if neighbors and wt_rand[i] < self.wt_growth_rate:
                # choose random neighbor
                rand_neighbor = np.random.randint(0,len(neighbors))
                # grow tumor
                grid[neighbors[rand_neighbor][0], neighbors[rand_neighbor][1]] = grid[wt_cells[0][i], wt_cells[1][i]]



        # grow mutant cells
        for i in range(len(mut_cells[0])):
            # check for neighbors
            mut_rand = np.random.rand(len(mut_cells[0]))
            # kill cells first
            if mut_rand[i] < self.mut_death_rate:
                grid[mut_cells[0][i], mut_cells[1][i]] = 0
            neighbors = self.check_neighbors(mut_cells[0][i], mut_cells[1][i], grid)
            # if neighbors and random number is less than growth rate, grow tumor
            if neighbors and mut_rand[i] < self.mut_growth_rate:
                # choose random neighbor
                rand_neighbor = np.random.randint(0,len(neighbors))
                # grow tumor
                grid[neighbors[rand_neighbor][0], neighbors[rand_neighbor][1]] = grid[mut_cells[0][i], mut_cells[1][i]]



        return grid

    def apply_treatment_action(self, action):
        if action == 0:
            self.wt_death_rate = self.reference_wt_death_rate
            self.mut_death_rate = self.reference_mut_death_rate
        elif action == 1:
            self.wt_death_rate = self.wt_drug_death_rate
            self.mut_death_rate = self.reference_mut_death_rate
        return


    def check_done(self, state):
        # check if done
        if np.sum(state) > self.max_tumor_size or self.time >= self.max_time:
            return True
        else:
            return False

    def check_neighbors(self, x, y, grid):
        # check for neighbors
        neighbors = []
        # check for neighbors
        if x > 0:
            if grid[x-1,y] == 0:
                neighbors.append([x-1,y])
        if x < self.grid_size-1:
            if grid[x+1,y] == 0:
                neighbors.append([x+1,y])
        if y > 0:
            if grid[x,y-1] == 0:
                neighbors.append([x,y-1])
        if y < self.grid_size-1:
            if grid[x,y+1] == 0:
                neighbors.append([x,y+1])
        return neighbors


    def reset(self):
        # reset time
        self.time = 0
        # reset state
        self.grid = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.uint8)
        self.place_cells(positioning=self.positioning)
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

    def render(self, mode='human'):
        # render state
        # plot it on the grid with different colors for wt and mut
        # animate simulation with matplotlib animation
        ims = []
        #color_map = mpl.colors.ListedColormap(['white', 'blue', 'yellow', 'red', 'green'])
        for i in range(self.time):
            im = self.ax.imshow(self.trajectory[:,:,i], animated=True)
            ims.append([im])
        ani = animation.ArtistAnimation(self.fig, ims, interval=5, blit=True, repeat_delay=1000)

        return ani



    def close(self):
        pass

if __name__ == "__main__":
    env = GridEnv()
    print(env.action_space)
    env.reset()
    #env.render()
    for i in range(200):
        env.step(0)
    for i in range(50):
        env.step(1)


    anim = env.render()
    env.close()
    plt.show()
    check_env(env, warn=True)

