# imports
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import os
import subprocess
import time

# create environment
class PC_env(Env):
    def __init__(self):
        # setting up environment
        # set up discrete action space
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0,high=100,shape=(3,))

        # set up timer
        self.time = 0
        self.max_time = 1000
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

    def step(self, action):
        # update tiemr
        self.time += 1
        # get tumor updated state

        if self.simulation_progressed(self.time):
            self.state[0] = get_PC_output()['susceptible']
            self.state[1] = get_PC_output()['resistant']
        else:
            q = 0
            while q < 1000:
                please_wait()
                if self.simulation_progressed(self.time):
                    break
                else:
                    q+=1
            if q == 1000:
                raise FileNotFoundError(' Simulation did not progress to this point ')
            self.state[0] = get_PC_output()['susceptible']
            self.state[1] = get_PC_output()['resistant']

        # do action (apply treatment or not)
        self.state[2] = action

        # record trajectory
        self.trajectory[:,self.time - 1] = self.state
        # get the reward
        reward = self.time/np.sum(self.state[0:2])
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
        self.start_new_sim()
        return self.state

    def start_new_sim(self):
        command = "conda deactivate && cd ..\PhysiCell_V_1.10.4 && project.exe && exit"
        p = subprocess.Popen(["start", "cmd", "/K", command], shell=True)
        if p.poll() is None:
            print("Running PhysiCell")
        return

    def cleanup(self):
        command = "conda deactivate && cd ..\PhysiCell_V_1.10.4 && make data-cleanup && exit"
        subprocess.call(["start","/wait", "cmd", "/K", command], shell=True)
        print("cleaned output")
        return

    def simulation_progressed(self,time):
        filename = 'output'+'{:04n}'.format(time)
        return True

if __name__ == '__main__':
    env = PC_env()
    env.reset()
    #command = "conda deactivate && cd ..\PhysiCell_V_1.10.4 && project.exe && exit"
    #p = subprocess.Popen(["start", "cmd", "/K", command], shell=True)
    #print("executing")
    #print(p.poll())




