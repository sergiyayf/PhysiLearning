# imports
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import os
import subprocess
from pathlib import Path
from get_output import get_PC_output
from treats import Treatment
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
        self.max_time = 4
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

    def step(self, action):
        # update timer
        self.time += 1
        # get tumor updated state

        if self.simulation_progressed(self.time):
            dict = get_PC_output(file = 'output'+'{:08n}'.format(self.time)+'.xml',
                                          dir = './../PhysiCell_V_1.10.4/output')
            self.state[0] = dict['susceptible']
            self.state[1] = dict['resistant']
        else:
            q = 0
            while q < 1000:
                # sleep x seconds
                time.sleep(5)
                # check for progress again
                if self.simulation_progressed(self.time):
                    # quit while loop
                    break
                else:
                    print('Waiting for progress')
                    q+=1
                if q == 1000:
                    raise FileNotFoundError(' Simulation did not progress to this point ')
            dict = get_PC_output(file='output' + '{:08n}'.format(self.time) + '.xml',
                                 dir='./../PhysiCell_V_1.10.4/output')
            self.state[0] = dict['susceptible']
            self.state[1] = dict['resistant']

        # do action (apply treatment or not)
        self.state[2] = action
        self.treatment.change_treatment(self.time,action)

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
        self.treatment.set_treatment_file_for_current_sim()
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




