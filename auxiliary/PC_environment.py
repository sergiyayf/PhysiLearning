# imports
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import os
import subprocess
from pathlib import Path
from get_output import get_PC_output
from treats import Treatment
import zmq

# create environment
class PC_env(Env):
    def __init__(self):
        # setting up environment
        # set up discrete action space
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0,high=100,shape=(3,))

        # set up timer
        self.time = 0
        self.max_time = 180
        self.treatment_time_step = 60

        # set up initial wild type, mutant and treatment decision
        self.initial_wt = 30
        self.initial_mut = 10
        self.initial_drug = 0

        # set up initial state
        self.state = [self.initial_wt,
                      self.initial_mut,
                      self.initial_drug]

        # trajectory for plotting
        self.trajectory = np.zeros((np.shape(self.state)[0],self.max_time))

        # treatment object
        self.treatment = Treatment()

        self.context = zmq.Context()
        self.socket = None

    def step(self, action):
        # update timer
        self.time += 1
        # get tumor updated state
        message = self.socket.recv()
        type0 = re.findall(r'%s(\d+)' % "Type 0:", message)
        self.state[0] = int(type0[0])
        type1 = re.findall(r'%s(\d+)' % "Type 1:", message)
        self.state[1] = int(type1[0])

        # do action (apply treatment or not)
        self.state[2] = action
        if action == 0 :
            self.socket.send(b"Stop treatment")
        elif action == 1:
            self.socket.send(b"Treat")
        # record trajectory
        self.trajectory[:,self.time - 1] = self.state
        # get the reward
        if np.sum(self.state[0:2]) != 0:
            reward = 100/np.sum(self.state[0:2])
        else:
            reward = 200;
        # check if we are done
        # check for existence of final.svg
        if self.time >= self.max_time:
            done = True
        else:
            done = False
        info = {}


        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.state = [self.initial_wt, self.initial_mut, self.initial_drug]

        self.time = 0

        self.trajectory = np.zeros((np.shape(self.state)[0],self.max_time))

        self.cleanup()

        # create a socket to talk to server
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("rcp://localhost:5555")
        # send initial no treatment request
        self.socket.send(b"Start simulation")
        self.start_new_sim()

        return self.state

    def start_new_sim(self):
        command = "conda deactivate && cd ..\PhysiCell_V_1.10.4 && project.exe && exit"
        p = subprocess.Popen(["start", "cmd", "/K", command], shell=True)
        if p.poll() is None:
            print("Running PhysiCell")
        return

    def cleanup(self):
        command = "cd ../PhysiCell_V_1.10.4 && make data-cleanup && exit"
        subprocess.call([command], shell=True)
        print("cleaned output")
        return

    def simulation_progressed(self,time):
        # check for existence of file with filename of this time
        filename = 'output'+'{:08n}'.format(time)+'.xml'
        directory = './../PhysiCell_V_1.10.4/output'
        file = Path(directory) / filename
        if os.path.exists(file):
            return True
        else:
            return False


if __name__ == '__main__':
    env = PC_env()
    env.reset()
    print(env.time)
    env.step(1)
    print(env.time)
    env.step(0)
    print(env.time)
    env.step(1)
    print(env.time)
    env.step(0)
    print(env.time)
    #command = "conda deactivate && cd ..\PhysiCell_V_1.10.4 && project.exe && exit"
    #p = subprocess.Popen(["start", "cmd", "/K", command], shell=True)
    #print("executing")
    #print(p.poll())




