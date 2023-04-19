import pytest
from physilearning import evaluate
import importlib
import numpy as np


@pytest.mark.parametrize('env_type', ['PcEnv', 'LvEnv', 'GridEnv'])
def test_at_fixed(env_type):
    """
    Test fixed adaptive therapy evaluation
    """
    EnvClass = getattr(importlib.import_module('physilearning.envs'), env_type)
    env = EnvClass(normalize_to=10, initial_wt=3, initial_mut=1)
    obs = np.array([4, 5])

    assert evaluate.fixed_at(obs, env, threshold=0.8, at_type='fixed')
    assert not evaluate.fixed_at(obs, env, threshold=0.95, at_type='fixed')
    assert evaluate.fixed_at(obs, env, at_type='mtd')
    assert evaluate.fixed_at(obs, env, at_type='zhang_et_al')
