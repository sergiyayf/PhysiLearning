import os

from physilearning.envs.base_env import BaseEnv
import numpy as np
import subprocess
import zmq
import re
import time
from physilearning.reward import Reward
import platform
from physilearning.tools.xml_reader import CfgRead


class PcEnv(BaseEnv):
    """
    PhysiCell environment

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
        name='PcEnv',
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
        port: str = '0',
        job_name: str = '0000000',
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

        # PhysiCell specific
        self.domain_size = env_specific_params.get('domain_size', 1250)
        self.job_name = job_name
        self.port = port
        if self.config['env']['patient_sampling']['enable']:
            self._get_patient_chkpt_file(self.patient_id)
        self.transport_type = env_specific_params.get('transport_type', 'ipc://')
        self.transport_address = env_specific_params.get('transport_address', f'/tmp/') + f'{self.job_name}{self.port}'
        self._bind_socket()
        # reward shaping flag
        self.cpu_per_task = env_specific_params.get('cpus_per_sim', 10)
        self.running = False
        self._start_slurm_physicell_job_step()


    def _bind_socket(self) -> None:
        """
        Bind the socket for communication between PhysiCell and the python environment
        Using ZMQ Request-Reply pattern. Can use ipc transport or potentially also tcp for remote execution
        or for Windows.
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        if self.transport_type == 'ipc://':
            self.socket.bind(f'{self.transport_type}{self.transport_address}')
        elif self.transport_type == 'tcp://':
            try:
                self.socket.bind(f'{self.transport_type}localhost:{self.transport_address}')
            except zmq.error.ZMQError:
                print("Connection failed. Double check the transport type and address. Trying with the default address")
                self.socket.bind(f'{self.transport_type}localhost:5555')
                self.transport_address = '5555'

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
            pc_cpus_per_task = self.cpu_per_task
            command = f"srun --ntasks=1 --exclusive --mem-per-cpu=200 " \
                      f"--cpus-per-task={pc_cpus_per_task} --cpu-bind=no ./scripts/run.sh {self.port} {port_connection}"
            #command = f"bash ./scripts/run.sh {self.port} {port_connection}"
            subprocess.Popen([command], shell=True)
        self.running = True
        self._receive_message()
        self._send_message('Start simulation')

    def _rewrite_xml_parameter(self, parent_nodes: list, parameter: str, value: str) -> None:
        """
        Rewrite a parameter in the PhysiCell_settings.xml file

        param: parent_nodes: list of parent nodes of the parameter to be rewritten
        param: parameter: parameter to be rewritten
        param: value: new value of the parameter
        """
        xml_reader = CfgRead(f'./simulations/PhysiCell_{self.port}/config/PhysiCell_settings.xml')
        xml_reader.write_new_param(parent_nodes=parent_nodes, parameter=parameter, value=value)

    def _get_patient_chkpt_file(self, patient_id):
        """
        Copy patient checkpoint file to the PhysiCell xml config

        """
        parameter = 'filename_chkpt'
        if self.config['env']['patient_sampling']['type'] == 'range':
            value = f'./../paper_presims/patient_{patient_id}/final'
        else:
            value = self.config['patients'][patient_id]['PcEnv']['filename_chkpt']['value']
        self._rewrite_xml_parameter(parent_nodes=['user_parameters'], parameter=parameter,
                                    value=value)

    def _send_message(self, message: str) -> None:
        """
        Send a message to the PhysiCell simulation

        param: message: message to be sent to the simulation
        """
        self.socket.send(bytes(message, 'utf-8'))

    def _receive_message(self) -> str:
        """
        Receive a message from the PhysiCell simulation

        return: message received from the simulation
        """
        return str(self.socket.recv(), 'utf-8')

    def step(self, action: int) -> tuple:
        """
        Receive a message from the PhysiCell simulation and
        send the action to the simulation

        param: action: value of the action to be sent to the simulation
        return: observation, reward, done, info
        """

        # check if the simulation is running or not, if not start physicell for testing
        if not self.running:
            self._start_slurm_physicell_job_step()

        if action == 0:
            self.socket.send(b"Stop treatment")
        elif action == 1:
            self.socket.send(b"Treat")

        self.time += self.treatment_time_step
        # get tumor updated state
        message = self._receive_message()
        if self.observation_type == 'image' or self.observation_type == 'multiobs':
            num_wt_cells, num_mut_cells = self._get_cell_number(message)
        elif self.observation_type == 'number':
            num_wt_cells, num_mut_cells = self._get_cell_number(message)
        else:
            raise ValueError('Observation type not supported')
        # num_wt_cells, num_mut_cells = self._get_cell_number(message)

        if self.normalize:
            self.state[0] = num_wt_cells * self.normalization_factor
            self.state[1] = num_mut_cells * self.normalization_factor
        else:
            self.state[0] = num_wt_cells
            self.state[1] = num_mut_cells

        self.state[2] = action
        # get from the string comma separated values from t0_x to t0_y
        if self.observation_type == 'image' or self.observation_type == 'multiobs':
            self.image = self._get_image_obs(message, action)
            self.image_trajectory[:, :, int(self.time/self.treatment_time_step)] = self.image[0, :, :]

            done = self._check_done(burden_type='number', total_cell_number=self.state[0] + self.state[1],
                                    message=message)
            self.trajectory[:, int(self.time/self.treatment_time_step)] = self.state
            rewards = Reward(self.reward_shaping_flag)
            reward = rewards.get_reward(self.state, self.time/self.max_time)

            # if done:
            #     print('Done')
            #     self.socket.send(b"End simulation")
            #     self.socket.close()
            #     self.context.term()
            # else:
            #     if action == 0:
            #         self.socket.send(b"Stop treatment")
            #     elif action == 1:
            #         self.socket.send(b"Treat")

            if self.observation_type == 'image':
                obs = self.image
            elif self.observation_type == 'multiobs':
                obs = {'vec': self.state, 'img': self.image}
            else:
                raise ValueError('Observation type not supported')

        elif self.observation_type == 'number':
            # record trajectory
            self.trajectory[:, int(self.time/self.treatment_time_step)] = self.state
            # get the reward
            rewards = Reward(self.reward_shaping_flag)
            reward = rewards.get_reward(self.state, self.time/self.max_time)

            obs = self.state

            # if self.time >= self.max_time or np.sum(self.state[0:2]) >= self.threshold_burden:
            #     done = True
            #     self.socket.send(b"End simulation")
            #     self.socket.close()
            #     self.context.term()
            #
            # else:
            #     done = False
            #     if action == 0:
            #         self.socket.send(b"Stop treatment")
            #     elif action == 1:
            #         self.socket.send(b"Treat")

        else:
            raise ValueError('Observation type not supported')
        info = {}
        terminate = self.terminate()
        truncate = self.truncate()
        if terminate or truncate:
            self.socket.send(b"End simulation")
            self.socket.close()
            self.context.term()
        return obs, reward, terminate, truncate, info

    def reset(self, *, seed=None, options=None):

        if self.config['env']['patient_sampling']['enable']:
            if len(self.patient_id_list) > 1:
                self._choose_new_patient()
                self._get_patient_chkpt_file(self.patient_id)

        if not self.running:
            self._start_slurm_physicell_job_step()
        self._bind_socket()
        _message = self._receive_message()
        self._send_message('Reset')

        message = self._receive_message()
        self.initial_wt, self.initial_mut = self._get_cell_number(message)

        if self.normalize:
            self.initial_wt *= self.normalization_factor
            self.initial_mut *= self.normalization_factor

        # self._send_message('Start simulation')

        self.state = [self.initial_wt, self.initial_mut, self.initial_drug]
        self.time = 0
        self.image = self._get_image_obs(message, 0)
        if self.observation_type == 'number':
            obs = self.state
            self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time / self.treatment_time_step)+1))
            self.trajectory[:, 0] = self.state
        elif self.observation_type == 'image' or self.observation_type == 'multiobs':
            self.image_trajectory = np.zeros(
                (self.image_size, self.image_size, int(self.max_time / self.treatment_time_step)+1))
            self.image_trajectory[:, :, 0] = self.image[0, :, :]
            self.trajectory = np.zeros(
                (np.shape(self.state)[0], int(self.max_time / self.treatment_time_step)+1))
            self.trajectory[:, 0] = self.state
            if self.observation_type == 'image':
                obs = self.image
            elif self.observation_type == 'multiobs':
                obs = {'vec': self.state, 'img': self.image}
            else:
                raise ValueError('Observation type not supported')
        else:
            raise ValueError('Observation type not supported')

        return obs, {}

    def _check_done(self, burden_type: str, **kwargs) -> bool:
        """
        Check if the episode is done: if the tumor is too big or the time is too long
        :param burden_type: type of burden to check
        :return: if the episode is done
        """

        if burden_type == 'number':
            num_wt_cells, num_mut_cells = self._get_cell_number(kwargs['message'])
            total_cell_number = num_wt_cells + num_mut_cells
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


if __name__ == '__main__':
    os.chdir("/home/saif/Projects/PhysiLearning")
    np.random.seed(15)
    env = PcEnv.from_yaml('./config.yaml', port='0', job_name='00000')
    print(env.patient_id)
    # env.reset()
    grid = env.image
    i = 0
    while i < 10:
        act = 0  # env.action_space.sample()
        env.step(act)
        i += 1

    anim = env.render()
