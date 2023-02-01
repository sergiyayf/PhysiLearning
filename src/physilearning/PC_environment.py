# imports
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

# create environment
class PC_env(Env):
    def __init__(self,port='0', job_name='0000000', burden=1000, max_time=30000,
            initial_wt=45, initial_mut=5, treatment_time_step=60, transport_type='ipc://',transport_address=f'/tmp/0',reward_shaping_flag=0):
        # setting up environment
        # set up discrete action space
        self.burden = burden 
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0,high=1,shape=(3,))

        # set up timer
        self.time = 0
        self.max_time = max_time
        self.treatment_time_step = treatment_time_step

        # set up initial wild type, mutant and treatment decision
        self.initial_wt = initial_wt
        self.initial_mut = initial_mut
        self.initial_drug = 0

        # set up initial state
        self.state = [self.initial_wt/self.burden,
                      self.initial_mut/self.burden,
                      self.initial_drug]

        # trajectory for plotting
        self.trajectory = np.zeros((np.shape(self.state)[0],int(self.max_time/self.treatment_time_step)))

        # socket
        self.job_name = job_name
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.transport_type = transport_type
        self.transport_address = transport_address
        self.socket.connect(f'{self.transport_type}{self.transport_address}')
        
        # reward shaping flag
        self.reward_shaping_flag = reward_shaping_flag

    @classmethod
    def from_yaml(cls,yaml_file,port='0',job_name = '000000'):
        with open(yaml_file,'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        burden = config['learning']['env']['burden']
        max_time = config['learning']['env']['max_time']
        initial_wt = config['learning']['env']['initial_wt']
        timestep = config['learning']['env']['treatment_time_step']
        initial_mut = config['learning']['env']['initial_mut']
        reward_shaping_flag = config['learning']['env']['reward_shaping']
        transport_type = config['global']['transport_type']
        transport_address = config['global']['transport_address']
        if transport_type == 'ipc://':
            transport_address = f'{transport_address}{job_name}{port}'
        else :
            warnings.warn('Transport type is different from ipc, please check the config file if everything is correct')
            transport_address = f'{transport_address}:{port}'
        
        return cls(port=port, job_name=job_name, burden=burden, max_time=max_time,
                initial_wt=initial_wt, treatment_time_step=timestep, initial_mut=initial_mut, transport_type=transport_type,
                transport_address=transport_address, reward_shaping_flag=reward_shaping_flag)


    def step(self, action):
        # update timer
        self.time += self.treatment_time_step
        # get tumor updated state
        
        message = str(self.socket.recv(),'utf-8')
         
        type0 = re.findall(r'%s(\d+)' % "Type 0:", message)
        self.state[0] = int(type0[0])/self.burden
        type1 = re.findall(r'%s(\d+)' % "Type 1:", message)
        self.state[1] = int(type1[0])/self.burden
        # do action (apply treatment or not)
        self.state[2] = action
        
        # record trajectory
        self.trajectory[:,int(self.time/self.treatment_time_step) - 1] = self.state
        # get the reward
        rewards = Reward(self.reward_shaping_flag)
        reward = rewards.get_reward(self.state)

        if self.time >= self.max_time or np.sum(self.state[0:2])>=1:
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

        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        time.sleep(3.0)
        port_connection = f"{self.transport_type}{self.transport_address}"
        command = f"bash ./bin/run.sh {self.port} {port_connection}"
        p = subprocess.Popen([command], shell=True)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f'{self.transport_type}{self.transport_address}')
        self.state = [self.initial_wt/self.burden, self.initial_mut/self.burden, self.initial_drug]
        self.time = 0
        self.trajectory = np.zeros((np.shape(self.state)[0],int(self.max_time/self.treatment_time_step)))
           
        self.socket.send(b"Start simulation")
        return self.state

    
if __name__ == '__main__':
    env = PC_env.from_yaml('./../../config.yaml')
    #env.reset()
    print(env.reward_shaping_flag)




