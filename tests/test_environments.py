# Test of the environments
import importlib
import pytest
from physilearning.envs import PcEnv, LvEnv, GridEnv
import yaml


@pytest.mark.parametrize('env_type', ['PcEnv', 'LvEnv', 'GridEnv'])
def test_env_creation(env_type):
    """
    Test environment creation.

    :param env_type: Environment type
    """
    if env_type == 'PcEnv':
        EnvClass = getattr(importlib.import_module('physilearning.envs.pc'), 'PcEnv')
    elif env_type == 'LvEnv':
        EnvClass = getattr(importlib.import_module('physilearning.envs.lv'), 'LvEnv')
    elif env_type == 'GridEnv':
        EnvClass = getattr(importlib.import_module('physilearning.envs.grid_env'), 'GridEnv')
    else:
        raise ValueError('Environment type not recognized')
    env = EnvClass()
    assert env is not None


@pytest.mark.parametrize('env_type', ['PcEnv', 'LvEnv', 'GridEnv'])
def test_env_reset(env_type):
    """
    Test environment reset.

    :param env_type: Environment type
    """
    if env_type == 'PcEnv':
        EnvClass = getattr(importlib.import_module('physilearning.envs.pc'), 'PcEnv')
    elif env_type == 'LvEnv':
        EnvClass = getattr(importlib.import_module('physilearning.envs.lv'), 'LvEnv')
    elif env_type == 'GridEnv':
        EnvClass = getattr(importlib.import_module('physilearning.envs.grid_env'), 'GridEnv')
    else:
        raise ValueError('Environment type not recognized')
    env = EnvClass()
    obs = env.reset()
    assert obs is not None

@pytest.mark.parametrize('env_type', ['PcEnv', 'LvEnv', 'GridEnv'])
def test_env_step(env_type):
    """
    Test environment step.

    :param env_type: Environment type
    """
    if env_type == 'PcEnv':
        EnvClass = getattr(importlib.import_module('physilearning.envs.pc'), 'PcEnv')
    elif env_type == 'LvEnv':
        EnvClass = getattr(importlib.import_module('physilearning.envs.lv'), 'LvEnv')
    elif env_type == 'GridEnv':
        EnvClass = getattr(importlib.import_module('physilearning.envs.grid_env'), 'GridEnv')
    else:
        raise ValueError('Environment type not recognized')
    env = EnvClass()
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    assert obs is not None
    assert reward is not None
    assert done is not None


def test_pc_env_from_yaml():
    """
    Test environment creation from yaml file.
    """
    with open('test_cfg.yaml', 'r') as f:
        config = yaml.safe_load(f)
    env = PcEnv.from_yaml('test_cfg.yaml')
    assert env is not None

    assert env.initial_wt == config['envs']['number_of_susceptible_cells']['value']


def test_grid_env_from_yaml():
    """
    Test environment creation from yaml file.
    """
    with open('test_cfg.yaml', 'r') as f:
        config = yaml.safe_load(f)
    env = GridEnv.from_yaml('test_cfg.yaml')
    assert env is not None


