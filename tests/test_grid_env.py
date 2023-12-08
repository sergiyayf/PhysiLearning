import importlib
import pytest
from physilearning.envs import PcEnv, GridEnv
import yaml


@pytest.mark.parametrize('env_type', ['LvEnv', 'GridEnv'])
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


@pytest.mark.parametrize('env_type', ['LvEnv', 'GridEnv'])
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


@pytest.mark.parametrize('env_type', ['LvEnv', 'GridEnv'])
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
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    assert obs is not None
    assert reward is not None
    assert done is not None
    assert trunc is not None


@pytest.mark.skip(reason="runs physicell")
def test_pc_env_from_yaml():
    """
    Test environment creation from yaml file.
    """
    with open('./tests/test_cfg.yaml', 'r') as f:
        config = yaml.safe_load(f)
    env = PcEnv.from_yaml('./tests/test_cfg.yaml')
    assert env is not None

    assert env.initial_wt == config['env']['PC']['number_of_susceptible_cells']['value']


def test_grid_env_from_yaml():
    """
    Test environment creation from yaml file.
    """
    env = GridEnv.from_yaml('./tests/test_cfg.yaml')
    assert env is not None

def test_grid_env_surround_mutant_placing():
    env = GridEnv()
    env.cell_positioning = 'surround_mutant'
    env.reset()
    mut_color = env.mut_color
    center = [env.image_size//2, env.image_size//2]
    assert env.image[0, center[0], center[1]] == mut_color

