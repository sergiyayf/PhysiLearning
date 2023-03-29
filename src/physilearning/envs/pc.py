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
        port: str = '0',
        job_name: str = '0000000',
        burden: float = 1000,
        max_time: int = 30000,
        initial_wt: int = 45,
        initial_mut: int = 5,
        treatment_time_step: int = 60,
        transport_type: str = 'ipc://',
        transport_address: str = f'/tmp/0',
        reward_shaping_flag: int = 0,
        normalize_to: float = 1000
    ) -> None:
        # Space
        self.name = 'PcEnv'
        self.threshold_burden_in_number = burden
        self.threshold_burden = normalize_to
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0,high=1,shape=(1,))

        # Timer
        self.time = 0
        self.max_time = max_time
        self.treatment_time_step = treatment_time_step

        # set up initial wild type, mutant and treatment decision
        self.initial_wt = initial_wt*self.threshold_burden/self.threshold_burden_in_number
        self.initial_mut = initial_mut*self.threshold_burden/self.threshold_burden_in_number
        self.initial_drug = 0

        # set up initial state
        self.state = [self.initial_wt,
                      self.initial_mut,
                      self.initial_drug]

        # trajectory for plotting
        self.trajectory = np.zeros((np.shape(self.state)[0],int(self.max_time/self.treatment_time_step)))

        # Socket
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
            except:
                print("Connection failed. Double check the transport type and address. Trying with the default address")
                self.socket.connect(f'{self.transport_type}localhost:5555')
                self.transport_address = '5555'


        # reward shaping flag
        self.reward_shaping_flag = reward_shaping_flag

    @classmethod
    def from_yaml(cls, yaml_file: str, port: str = '0', job_name: str = '000000') -> object:
        with open(yaml_file,'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        burden = config['env']['threshold_burden']
        normalize_to = config['env']['normalize_to']
        max_time = config['env']['max_time']
        initial_wt = config['env']['PC']['number_of_susceptible_cells']['value']
        timestep = config['env']['treatment_time_step']
        initial_mut = config['env']['PC']['number_of_resistant_cells']['value']
        reward_shaping_flag = config['env']['reward_shaping']
        transport_type = config['global']['transport_type']
        transport_address = config['global']['transport_address']
        if transport_type == 'ipc://':
            transport_address = f'{transport_address}{job_name}{port}'
        else :
            warnings.warn('Transport type is different from ipc, please check the config file if everything is correct')
            transport_address = f'{transport_address}:{port}'
        
        return cls(port=port, job_name=job_name, burden=burden, max_time=max_time,
                initial_wt=initial_wt, treatment_time_step=timestep, initial_mut=initial_mut, transport_type=transport_type,
                transport_address=transport_address, reward_shaping_flag=reward_shaping_flag, normalize_to=normalize_to)

    def step(self, action: int) -> tuple:
        # update timer
        self.time += self.treatment_time_step
        # get tumor updated state

        message = str(self.socket.recv(),'utf-8')
         
        type0 = re.findall(r'%s(\d+)' % "Type 0:", message)
        self.state[0] = int(type0[0])*self.threshold_burden/self.threshold_burden_in_number
        type1 = re.findall(r'%s(\d+)' % "Type 1:", message)
        self.state[1] = int(type1[0])*self.threshold_burden/self.threshold_burden_in_number
        # do action (apply treatment or not)
        self.state[2] = action

        # record trajectory
        self.trajectory[:,int(self.time/self.treatment_time_step) - 1] = self.state
        # get the reward
        rewards = Reward(self.reward_shaping_flag)
        reward = rewards.get_reward(self.state,self.time/self.max_time)

        if self.time >= self.max_time or np.sum(self.state[0:2])>=self.threshold_burden:
            done = True
            self.socket.send(b"End simulation")
            self.socket.close()
            self.context.term()
            
        else:
            done = False
            if action == 0 :
                self.socket.send(b"Stop treatment")
            elif action == 1:
                self.socket.send(b"Treat")
        
        info = {}

        return [np.sum(self.state[0:2])], reward, done, info

    def render(self):
        pass

    def reset(self):
        time.sleep(3.0)
        if self.transport_type == 'ipc://':
            port_connection = f"{self.transport_type}{self.transport_address}"
        elif self.transport_type == 'tcp://':
            port_connection = f"{self.transport_type}*:{self.transport_address}"

        if platform.system() == 'Windows':
            raise NotImplementedError('Windows is not supported yet')
            command = f"conda deactivate && bash ./scripts/run.sh {self.port} {port_connection}"
            p = subprocess.Popen(["start", "cmd", "/K", command], shell=True)

        else:
            command = f"bash ./scripts/run.sh {self.port} {port_connection}"
            p = subprocess.Popen([command], shell=True)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f'{self.transport_type}{self.transport_address}')
        self.state = [self.initial_wt, self.initial_mut, self.initial_drug]
        self.time = 0
        self.trajectory = np.zeros((np.shape(self.state)[0],int(self.max_time/self.treatment_time_step)))
           
        self.socket.send(b"Start simulation")
        return [np.sum(self.state[0:2])]

    
if __name__ == '__main__':
    env = PcEnv.from_yaml('../../../config.yaml')
    #env.reset()
    print(env.reward_shaping_flag)




