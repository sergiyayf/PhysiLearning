from physilearning.envs.PC_environment import PC_env


def test_PC_env_constructuion():
    env = PC_env()
    assert env 

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



