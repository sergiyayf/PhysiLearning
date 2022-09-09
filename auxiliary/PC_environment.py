# imports
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import os
import subprocess
from pathlib import Path
import zmq
import re
import time

# create environment
class PC_env(Env):
    def __init__(self,port):
        # setting up environment
        # set up discrete action space
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0,high=100,shape=(3,))

        # set up timer
        self.time = 0
        self.max_time = 30000
        self.treatment_time_step = 60

        # set up initial wild type, mutant and treatment decision
        self.initial_wt = 45
        self.initial_mut = 5
        self.initial_drug = 0

        # set up initial state
        self.state = [self.initial_wt,
                      self.initial_mut,
                      self.initial_drug]

        # trajectory for plotting
        self.trajectory = np.zeros((np.shape(self.state)[0],self.max_time))

        # socket
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect('ipc:///raven/ptmp/saif/'+self.port)
        
        # simulation 
        self.simulation = None

    def step(self, action):
        # update timer
        self.time += 60
        # get tumor updated state
        print('stepping')
        message = str(self.socket.recv(),'utf-8')
         
        type0 = re.findall(r'%s(\d+)' % "Type 0:", message)
        self.state[0] = int(type0[0])
        type1 = re.findall(r'%s(\d+)' % "Type 1:", message)
        self.state[1] = int(type1[0])
        # do action (apply treatment or not)
        self.state[2] = action
        
        # record trajectory
        self.trajectory[:,self.time - 1] = self.state
        # get the reward
        if np.sum(self.state[0:2]) != 0:
            reward = (1000-self.state[0]-5*self.state[1])*1e-3 # this means about 60 % of the simulation space is filled
        else:
            reward = 5
        # check if we are done
        #reward = 1 
        if self.time >= self.max_time or np.sum(self.state[0:2])>=1000:
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
        command = "cd ../PhysiCell_V_1.10.4_"+self.port+" && make data-cleanup && exit"
        subprocess.run([command], shell=True)

        command = "bash run.sh "+self.port
        p = subprocess.Popen([command], shell=True)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect('ipc:///raven/ptmp/saif/'+self.port)
        self.state = [self.initial_wt, self.initial_mut, self.initial_drug]
        self.time = 0
        self.trajectory = np.zeros((np.shape(self.state)[0],self.max_time))
            
        self.socket.send(b"Start simulation")
        print('Reset sim, bound to port'+self.port)
        return self.state

    
if __name__ == '__main__':
    env = PC_env()
    env.reset()
    print(env.time)
    
    #command = "conda deactivate && cd ..\PhysiCell_V_1.10.4 && project.exe && exit"
    #p = subprocess.Popen(["start", "cmd", "/K", command], shell=True)
    #print("executing")
    #print(p.poll())




