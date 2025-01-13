import os

import pandas as pd

from physilearning.envs.base_env import BaseEnv
import numpy as np
import subprocess
import zmq
import re
import time
from physilearning.reward import Reward
import platform
from physilearning.tools.xml_reader import CfgRead
from physicell_tools.get_perifery import front_cells
from physicell_tools.leastsquares import leastsq_circle


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
        see_resistance: bool = False,
        see_prev_action: bool = False,
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
        reward_shaping_flag: str = 'ttp',
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
                         see_resistance=see_resistance, see_prev_action=see_prev_action
                         )
        # check supported observation spaces
        if self.observation_type not in ['number', 'image', 'multiobs', 'mutant_position']:
            raise ValueError('Observation type not supported')
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
        #self._start_slurm_physicell_job_step()

        self.cell_df = pd.DataFrame()
        self.dimension = 2
        self.radius = 0
        self.mutant_radial_position = 0
        self.mutant_normalized_position = 0


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
            command = f"srun --ntasks=1 --exclusive --mem-per-cpu=100 " \
                      f"--cpus-per-task={pc_cpus_per_task} --cpu-bind=no ./scripts/run.sh {self.port} {port_connection}"
            # command = f"bash ../../../scripts/run.sh {self.port} {port_connection}"
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
            self.reset()
        reward = 0
        self.state[2] = action
        for t in range(0, self.treatment_time_step):

            if action == 0:
                self.socket.send(b"Stop treatment")
            elif action == 1:
                self.socket.send(b"Treat")

            self.time += 1
            # get tumor updated state
            message = self._receive_message()
            num_wt_cells, num_mut_cells = self._get_cell_number(message)

            if self.normalize:
                self.state[0] = num_wt_cells * self.normalization_factor
                self.state[1] = num_mut_cells * self.normalization_factor
            else:
                self.state[0] = num_wt_cells
                self.state[1] = num_mut_cells


            # get from the string comma separated values from t0_x to t0_y
            if self.observation_type == 'image' or self.observation_type == 'multiobs':
                self.image = self._get_image_obs(message, action)
                self.image_trajectory[:, :, int(self.time)] = self.image[0, :, :]
                self.trajectory[:, int(self.time)] = self.state

                if self.observation_type == 'image':
                    obs = self.image
                elif self.observation_type == 'multiobs':
                    obs = {'vec': self.state, 'img': self.image}
                else:
                    raise ValueError('Observation type not supported')

            elif self.observation_type == 'number':
                # record trajectory
                self.trajectory[:, int(self.time)] = self.state

                if self.see_resistance:
                    obs = self.state[0:2]
                else:
                    obs = [np.sum(self.state[0:2])]
                if self.see_prev_action:
                    obs = np.append(obs, action)

            elif self.observation_type == 'mutant_position':
                # measure tumor radius
                cell_df = self._get_df_from_message(message)
                self.radius = self._measure_radius()
                mutants_df = cell_df[cell_df['cell_type'] == 1]
                mut_dist_dict= self._calculate_distances_to_front(mutants_df)
                min_mut_dist = mut_dist_dict['min_front_distance']
                self.mutant_normalized_position = 1 - min_mut_dist / self.radius
                self.trajectory[0:3, int(self.time)] = self.state
                self.trajectory[3, int(self.time)] = self.mutant_normalized_position
                self.trajectory[4, int(self.time)] = self.radius
                if self.see_resistance:
                    obs = self.state
                    obs.append(self.mutant_normalized_position)
                else:
                    obs = [np.sum(self.state[0:2]), self.state[2], self.mutant_normalized_position]

            else:
                raise ValueError('Observation type not supported')

            info = {}
            reward += self.get_reward()

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
        initial_wt, initial_mut = self._get_cell_number(message)

        if self.normalize:
            self.normalization_factor = self.normalize_to / (initial_mut + initial_wt)
            self.threshold_burden = self.normalize_to*self.max_tumor_size
            self.initial_wt = initial_wt * self.normalization_factor
            self.initial_mut = initial_mut * self.normalization_factor

        else:
            self.threshold_burden = self.max_tumor_size * (initial_mut + initial_wt)
            self.initial_wt = initial_wt
            self.initial_mut = initial_mut
            self.normalization_factor = 1

        # self._send_message('Start simulation')

        self.state = [self.initial_wt, self.initial_mut, self.initial_drug]
        self.time = 0

        if self.observation_type == 'number':
            if self.see_resistance:
                obs = self.state[0:2]
            else:
                obs = [np.sum(self.state[0:2])]
            self.trajectory = np.zeros((np.shape(self.state)[0], int(self.max_time)+1))
            self.trajectory[:, 0] = self.state
        elif self.observation_type == 'image' or self.observation_type == 'multiobs':
            self.image = self._get_image_obs(message, action=0)
            self.image_trajectory = np.zeros(
                (self.image_size, self.image_size, int(self.max_time)+1))
            self.image_trajectory[:, :, 0] = self.image[0, :, :]
            self.trajectory = np.zeros(
                (np.shape(self.state)[0], int(self.max_time)+1))
            self.trajectory[:, 0] = self.state
            if self.observation_type == 'image':
                obs = self.image
            elif self.observation_type == 'multiobs':
                obs = {'vec': self.state, 'img': self.image}
            else:
                raise ValueError('Observation type not supported')
        elif self.observation_type == 'mutant_position':

            cell_df = self._get_df_from_message(message)
            self.radius = self._measure_radius()
            mutants_df = cell_df[cell_df['cell_type'] == 1]
            mut_dist_dict = self._calculate_distances_to_front(mutants_df)
            min_mut_dist = mut_dist_dict['min_front_distance']
            self.mutant_normalized_position = 1 - min_mut_dist / self.radius
            self.trajectory = np.zeros(
                (np.shape(self.state)[0]+2, int(self.max_time)+1))
            self.trajectory[0:3, 0] = self.state
            self.trajectory[3, 0] = self.mutant_normalized_position
            self.trajectory[4, 0] = self.radius
            if self.see_resistance:
                obs = self.state
                obs.append(self.mutant_normalized_position)
            else:
                obs = [np.sum(self.state[0:2]), self.state[2], self.mutant_normalized_position]
        else:
            raise ValueError('Observation type not supported')

        # not clean pulse hack
        # for tt in [0,1]:
        #    self._send_message('Stop treatment')
        #    message = self._receive_message()
        #    wt, mut = self._get_cell_number(message)
        #    wt = wt * self.normalization_factor
        #    mut = mut * self.normalization_factor
        #    self.state = [wt, mut, 0]
        #    self.time += 1
        #    self.trajectory[0:3, self.time] = self.state
        #self.threshold_burden = self.max_tumor_size * (self.state[0] + self.state[1])

        if self.see_resistance:
            obs = self.state[0:2]
        else:
            obs = [np.sum(self.state[0:2])]
        if self.see_prev_action:
            obs = np.append(obs, 0)
        return obs, {}

    def _measure_radius(self):
        """
        Measure the tumor radius. Try getting cells in convex hull and calculate average radius. If not possible,
        get the largest radial position of cells.
        """
        # get front cells if colony is large enough
        try:
            positions, types = front_cells(self.cell_df)
            self.cell_df.loc[self.cell_df['position_x'].isin(positions[:, 0]), 'is_at_front'] = 1
            #radius = np.mean(np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2 + positions[:, 2] ** 2))
            xc, yc, radius, res = leastsq_circle(positions[:, 0], positions[:, 1])
            # catch error if covex hull not possible get the largest radial position of cell.
        except:
            positions = self.cell_df[['position_x', 'position_y', 'position_z']].values
            radius = np.max(np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2 + positions[:, 2] ** 2))
        self.radius = radius

        return radius

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

    def _get_df_from_message(self, message) -> pd.DataFrame:
        t0_start_x = message.find('t0_x:') + len('t0_x:')
        t0_end_x = message.find('t0_y:')
        t0_x = message[t0_start_x:t0_end_x].split(',')
        t0_x = np.array([float(x) for x in t0_x[0:-1]])
        t0_start_y = message.find('t0_y:') + len('t0_y:')
        t0_end_y = message.find('t0_z:')
        t0_y = message[t0_start_y:t0_end_y].split(',')
        t0_y = np.array([float(y) for y in t0_y[0:-1]])
        t0_start_z = message.find('t0_z:') + len('t0_z:')
        t0_end_z = message.find('t1_x:')
        t0_z = message[t0_start_z:t0_end_z].split(',')
        t0_z = np.array([float(z) for z in t0_z[0:-1]])

        t1_start_x = message.find('t1_x:') + len('t1_x:')
        t1_end_x = message.find('t1_y:')
        t1_x = message[t1_start_x:t1_end_x].split(',')
        t1_x = np.array([float(x)  for x in t1_x[0:-1]])
        t1_start_y = message.find('t1_y:') + len('t1_y:')
        t1_end_y = message.find('t1_z:')
        t1_y = message[t1_start_y:t1_end_y].split(',')
        t1_y = np.array([float(y) for y in t1_y[0:-1]])
        t1_start_z = message.find('t1_z:') + len('t1_z:')
        t1_z = message[t1_start_z:].split(',')
        t1_z = np.array([float(z) for z in t1_z[0:-1]])

        # create data frame with positions and types of cells
        t0 = pd.DataFrame({'position_x': t0_x, 'position_y': t0_y, 'position_z': t0_z, 'cell_type': 0})
        t1 = pd.DataFrame({'position_x': t1_x, 'position_y': t1_y, 'position_z': t1_z, 'cell_type': 1})
        cells = pd.concat([t0, t1])
        # set unique indices
        cells.index = range(len(cells))
        self.cell_df = cells
        self.cell_df['is_at_front'] = np.zeros_like(self.cell_df['position_x'])
        return cells

    def _calculate_distances_to_front(self, dataframe):
        df = dataframe.copy()
        df.loc[:,'distance_to_center'] = (np.sqrt(
            (df['position_x']) ** 2 + (df['position_y']) ** 2 + (df['position_z']) ** 2)).values
        front_cell_positions = self.cell_df[self.cell_df['is_at_front'] == 1][['position_x', 'position_y', 'position_z']].values
        for i in range(len(front_cell_positions)):
            current_dist = np.sqrt(
                (df['position_x'] - front_cell_positions[i, 0]) ** 2 + (df['position_y'] - front_cell_positions[i, 1]) ** 2 + (
                            df['position_z'] - front_cell_positions[i, 2]) ** 2)
            if i == 0:
                dist = current_dist
            else:
                dist = np.minimum(dist, current_dist)
        df.loc[:,'distance_to_front_cell'] = dist
        # calculate average and minimum distances to front_
        avg_front_distance = np.mean(df['distance_to_front_cell'])
        min_front_distance = np.min(df['distance_to_front_cell'])
        avg_center_distance = np.mean(df['distance_to_center'])
        min_center_distance = np.min(df['distance_to_center'])
        dists = {'avg_front_distance': avg_front_distance, 'min_front_distance': min_front_distance,
                 'avg_center_distance': avg_center_distance, 'min_center_distance': min_center_distance}
        return dists

    def _get_image_obs(self, message: str, action: int) -> np.ndarray:
        """
        Get the image observation from the message received from the socket
        Look for the string ti_x: and ti_y: to get the coordinates of the type i cells

        :param message: message received from the PhysiCell simulation
        :return: image observation
        """
        if self.cell_df.empty:
            self.cell_df = self._get_df_from_message(message)
        t0_x = (self.cell_df[self.cell_df['cell_type'] == 0]['position_x'].values+self.domain_size/2)/self.domain_size*self.image_size
        t0_y = (self.cell_df[self.cell_df['cell_type'] == 0]['position_y'].values+self.domain_size/2)/self.domain_size*self.image_size
        t1_x = (self.cell_df[self.cell_df['cell_type'] == 1]['position_x'].values+self.domain_size/2)/self.domain_size*self.image_size
        t1_y = (self.cell_df[self.cell_df['cell_type'] == 1]['position_y'].values+self.domain_size/2)/self.domain_size*self.image_size

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


if __name__ == '__main__': # pragma: no cover
    os.chdir("/home/saif/Projects/PhysiLearning")
    np.random.seed(15)
    env = PcEnv.from_yaml('./config.yaml', port='0', job_name='00000')
    print(env.patient_id)
    # env.reset()
    grid = env.image
    i = 0
    env.reset()
    print('Normalized mutant position ',env.mutant_normalized_position)
    while i < 2:
        act = 0  # env.action_space.sample()
        obs, rew, term, trunc, info = env.step(act)
        i += 1
        print('Normalized mutant position ', env.mutant_normalized_position)
        print('Observation', obs)

    anim = env.render()
