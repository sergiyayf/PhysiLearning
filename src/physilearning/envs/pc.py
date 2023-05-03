from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import subprocess
import zmq
import re
import time
import yaml
import warnings
from physilearning.reward import Reward
import platform


class PcEnv(Env):
    """
    PhysiCell environment

    :param port: port number for zmq communication
    :param job_name: job name for zmq communication
    :param burden: burden threshold in number of cells
    :param max_time: maximum time steps
    :param initial_wt: initial number of wild type cells
    :param initial_mut: initial number of mutant cells
    :param treatment_time_step: time step at which treatment is applied
    :param transport_type: transport type for zmq communication
    :param transport_address: transport address for zmq communication
    :param reward_shaping_flag: flag to enable reward shaping
    :param normalize_to: normalization factor for reward shaping
    """
    def __init__(
        self,
        observation_type: str = 'image',
        action_type: str = 'discrete',
        image_size: int = 128,
        normalize: bool = False,
        normalize_to: float = 1000,
        max_tumor_size: float = 1000,
        max_time: int = 30000,
        treatment_time_step: int = 60,
        reward_shaping_flag: int = 0,
        initial_wt: int = 45,
        initial_mut: int = 5,
        transport_type: str = 'ipc://',
        transport_address: str = '/tmp/0',
        port: str = '0',
        job_name: str = '0000000',
    ) -> None:
        #################### Todo: move to base class ##################
        # Spaces
        self.name = 'PcEnv'
        self.action_type = action_type
        if self.action_type == 'discrete':
            self.action_space = Discrete(2)
        elif self.action_type == 'continuous':
            self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_type = observation_type
        if self.observation_type == 'number':
            self.observation_space = Box(low=0, high=1, shape=(1,))
        elif self.observation_type == 'image':
            self.observation_space = Box(low=0, high=255,
                                         shape=(1, image_size, image_size),
                                         dtype=np.uint8)
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

        # set up initial wild type, mutant and treatment decision
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
            self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time/self.treatment_time_step)))
        elif self.observation_type == 'image':
            self.trajectory = np.zeros((self.image_size, self.image_size, int(self.max_time/self.treatment_time_step)))
            self.number_trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time/self.treatment_time_step)))

        ######################################################
        # PhysiCell specific for now
        self.domain_size = 1000
        self.job_name = job_name
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.transport_type = transport_type
        self.transport_address = transport_address
        if transport_type == 'ipc://':
            self.socket.connect(f'{self.transport_type}{self.transport_address}')
        elif transport_type == 'tcp://':
            try:
                self.socket.connect(f'{self.transport_type}localhost:{self.transport_address}')
            except zmq.error.ZMQError:
                print("Connection failed. Double check the transport type and address. Trying with the default address")
                self.socket.connect(f'{self.transport_type}localhost:5555')
                self.transport_address = '5555'
        # reward shaping flag

        self._start_slurm_physicell_job_step()

    @classmethod
    def from_yaml(cls, yaml_file: str, port: str = '0', job_name: str = '000000') -> object:
        with open(yaml_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        max_tumor_size = config['env']['max_tumor_size']
        normalize_to = config['env']['normalize_to']
        max_time = config['env']['max_time']
        initial_wt = config['env']['PC']['number_of_susceptible_cells']['value']
        timestep = config['env']['treatment_time_step']
        initial_mut = config['env']['PC']['number_of_resistant_cells']['value']
        reward_shaping_flag = config['env']['reward_shaping']
        observation_type = config['env']['observation_type']
        image_size = config['env']['image_size']
        transport_type = config['global']['transport_type']
        transport_address = config['global']['transport_address']
        if transport_type == 'ipc://':
            transport_address = f'{transport_address}{job_name}{port}'
        else:
            warnings.warn('Transport type is different from ipc, please check the config file if everything is correct')
            transport_address = f'{transport_address}:{port}'
        
        return cls(port=port, job_name=job_name, max_tumor_size=max_tumor_size, max_time=max_time,
                   initial_wt=initial_wt, treatment_time_step=timestep, initial_mut=initial_mut,
                   transport_type=transport_type, transport_address=transport_address,
                   reward_shaping_flag=reward_shaping_flag, normalize_to=normalize_to,
                   observation_type=observation_type, image_size=image_size)

    def _start_slurm_physicell_job_step(self) -> None:
        """
        Start the PhysiCell simulation
        """
        if self.transport_type == 'ipc://':
            port_connection = f"{self.transport_type}{self.transport_address}"
        elif self.transport_type == 'tcp://':
            port_connection = f"{self.transport_type}*:{self.transport_address}"
        else:
            raise ValueError('Transport type not supported')

        if platform.system() == 'Windows':
            raise NotImplementedError('Windows is not supported yet')
            # command = f"conda deactivate && bash ./scripts/run.sh {self.port} {port_connection}"
            # p = subprocess.Popen(["start", "cmd", "/K", command], shell=True)

        else:
            pc_cpus_per_task = 1
            command = f"srun --ntasks=1 --exclusive --mem-per-cpu=300 " \
                      f"--cpus-per-task={pc_cpus_per_task} ./scripts/run.sh {self.port} {port_connection}"
            subprocess.Popen([command], shell=True)

    def step(self, action: int) -> tuple:
        """
        Receive a message from the PhysiCell simulation and
        send the action to the simulation

        param: action: value of the action to be sent to the simulation
        return: observation, reward, done, info
        """
        self.time += self.treatment_time_step
        # get tumor updated state
        message = str(self.socket.recv(), 'utf-8')
        im = self._get_image_obs(message, action)
        num_wt_cells, num_mut_cells = self._get_tumor_volume_from_image(im)
        # num_wt_cells, num_mut_cells = self._get_cell_number(message)

        if self.normalize:
            self.state[0] = num_wt_cells * self.normalization_factor
            self.state[1] = num_mut_cells * self.normalization_factor
        else:
            self.state[0] = num_wt_cells
            self.state[1] = num_mut_cells

        self.state[2] = action
        # get from the string comma separated values from t0_x to t0_y
        if self.observation_type == 'image':
            self.image = self._get_image_obs(message, action)
            self.trajectory[:, :, int(self.time/self.treatment_time_step) - 1] = self.image[0, :, :]
            obs = self.image
            done = self._check_done(burden_type='number', total_cell_number=num_wt_cells+num_mut_cells)
            self.number_trajectory[:, int(self.time/self.treatment_time_step) - 1] = self.state
            rewards = Reward(self.reward_shaping_flag)
            reward = rewards.get_reward(self.state, self.time/self.max_time)

            if done:
                print('Done')
                self.socket.send(b"End simulation")
                self.socket.close()
                self.context.term()
            else:
                if action == 0:
                    self.socket.send(b"Stop treatment")
                elif action == 1:
                    self.socket.send(b"Treat")

        elif self.observation_type == 'number':
            # record trajectory
            self.trajectory[:, int(self.time/self.treatment_time_step) - 1] = self.state
            # get the reward
            rewards = Reward(self.reward_shaping_flag)
            reward = rewards.get_reward(self.state, self.time/self.max_time)

            obs = self.state

            if self.time >= self.max_time or np.sum(self.state[0:2]) >= self.threshold_burden:
                done = True
                self.socket.send(b"End simulation")
                self.socket.close()
                self.context.term()

            else:
                done = False
                if action == 0:
                    self.socket.send(b"Stop treatment")
                elif action == 1:
                    self.socket.send(b"Treat")

        else:
            raise ValueError('Observation type not supported')
        info = {}

        return obs, reward, done, info

    def reset(self):
        time.sleep(3.0)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f'{self.transport_type}{self.transport_address}')
        self.state = [self.initial_wt, self.initial_mut, self.initial_drug]
        self.time = 0
        self.image = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)
        if self.observation_type == 'number':
            obs = [np.sum(self.state[0:2])]
            self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time / self.treatment_time_step)))
        elif self.observation_type == 'image':
            obs = self.image
            self.trajectory = np.zeros(
                (self.image_size, self.image_size, int(self.max_time / self.treatment_time_step)))
            self.number_trajectory = np.zeros(
                (np.shape(self.state)[0], int(self.max_time / self.treatment_time_step)))
        else:
            raise ValueError('Observation type not supported')

        self.socket.send(b"Start simulation")
        return obs

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

    def _get_tumor_volume_from_image(self, state: np.ndarray) -> tuple:
        """
        Calculate the number of wt and mut cells in the state
        :param state: the state
        :return: number of wt and mut cells
        """
        num_wt_cells = np.sum(state == self.wt_color)
        num_mut_cells = np.sum(state == self.mut_color)
        return num_wt_cells, num_mut_cells

    @staticmethod
    def _get_cell_number(message: str) -> tuple:
        """
        Get the number of cells from the message received from the socket
        Look for the string Type 0: and Type 1: to get the number of cells

        :param message: message received from the PhysiCell simulation
        :return: number of wt and mut cells
        """
        type0 = re.findall(r'%s(\d+)' % "Type 0:", message)
        type1 = re.findall(r'%s(\d+)' % "Type 1:", message)
        return int(type0[0]), int(type1[0])

    def _get_image_obs(self, message: str, action: int) -> np.ndarray:
        """
        Get the image observation from the message received from the socket
        Look for the string ti_x: and ti_y: to get the coordinates of the type i cells

        :param message: message received from the PhysiCell simulation
        :return: image observation
        """
        t0_start_index = message.find('t0_x:') + len('t0_x:')
        t0_end_index = message.find('t0_y:')
        t0_x = message[t0_start_index:t0_end_index].split(',')
        t0_x = np.array([float(x) + self.domain_size / 2 for x in t0_x[0:-1]])
        t0_y = message[t0_end_index + len('t0_y:'):message.find('t1_x:')].split(',')
        t0_y = np.array([float(y) + self.domain_size / 2 for y in t0_y[0:-1]])

        t1_start_index = message.find('t1_x:') + len('t1_x:')
        t1_end_index = message.find('t1_y:')
        t1_x = message[t1_start_index:t1_end_index].split(',')
        t1_x = np.array([float(x) + self.domain_size / 2 for x in t1_x[0:-1]])
        t1_y = message[t1_end_index + len('t1_y:'):].split(',')
        t1_y = np.array([float(y) + self.domain_size / 2 for y in t1_y[0:-1]])

        # normalize the coordinates to the image size
        t0_x = np.round(t0_x * self.image_size / self.domain_size)
        t0_y = np.round(t0_y * self.image_size / self.domain_size)
        t1_x = np.round(t1_x * self.image_size / self.domain_size)
        t1_y = np.round(t1_y * self.image_size / self.domain_size)
        # clean the image and make the new one
        if action:
            self.image = self.drug_color*np.ones((1, self.image_size, self.image_size), dtype=np.uint8)
        else:
            self.image = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)

        for x, y in zip(t0_x, t0_y):
            self.image[0, int(x), int(y)] = self.wt_color
        for x, y in zip(t1_x, t1_y):
            self.image[0, int(x), int(y)] = self.mut_color

        return self.image


def render(trajectory: np.ndarray, timestep: int, fig, ax):
    # render state
    # plot it on the grid with different colors for wt and mut
    # animate simulation with matplotlib animation
    import matplotlib.animation as animation
    ims = []
    for i in range(timestep):
        im = ax.imshow(trajectory[:, :, i], animated=True, cmap='viridis', vmin=0, vmax=255)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=0.1, blit=True, repeat_delay=1000)

    return ani


if __name__ == '__main__':
    env = PcEnv.from_yaml('../../../config.yaml')
    # env.reset()
