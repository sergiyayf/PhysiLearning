import stable_baselines3
from physilearning.PC_environment import PC_env
from physilearning.ODE_environments import LV_env
import pytest

def test_PC_env_constructuion():
    env = PC_env()
    assert env 

def test_LV_env_constr

def test_PC_env_from_yaml():
    env = PC_env.from_yaml('../config.yaml')
    assert env

def test_reset_env():
    #env = PC_env()
    #env.reset()
    assert True 

def test_env_step(): 
    #env = PC_env()
    #env.reset()
    #obs = env.step()
    assert True



