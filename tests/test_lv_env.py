from physilearning.envs import LvEnv
import pytest
import numpy as np


def test_observation_space():
    env = LvEnv(observation_type='number')
    assert env.observation_type == 'number'

    with pytest.raises(NotImplementedError):
        env = LvEnv(observation_type='image')


def test_random_cell_number():
    env = LvEnv(initial_wt=5)
    assert env.initial_wt == 5
    env.reset()
    assert env.state[0] == 5

    initials_wt = []
    initials_mut = []
    for i in range(10):
        env = LvEnv(initial_wt='random', initial_mut='random')
        initial_wt = env.initial_wt
        initial_mut = env.initial_mut
        initials_wt.append(initial_wt)
        initials_mut.append(initial_mut)

    diff_wt = [initials_wt[i] - initials_wt[i+1] for i in range(len(initials_wt)-1)]
    diff_mut = [initials_mut[i] - initials_mut[i+1] for i in range(len(initials_mut)-1)]
    assert np.sum(diff_wt) != 0
    assert np.sum(diff_mut) != 0


def test_random_cell_number_with_reset():
    initials_wt = []
    initials_mut = []
    for i in range(10):
        env = LvEnv(initial_wt='random', initial_mut='random')
        env.reset()
        initial_wt = env.initial_wt
        initial_mut = env.initial_mut
        initials_wt.append(initial_wt)
        initials_mut.append(initial_mut)

    diff_wt = [initials_wt[i] - initials_wt[i+1] for i in range(len(initials_wt)-1)]
    diff_mut = [initials_mut[i] - initials_mut[i+1] for i in range(len(initials_mut)-1)]
    assert np.sum(diff_wt) != 0
    assert np.sum(diff_mut) != 0

def test_normalization():
    env = LvEnv(normalize=True, normalize_to=1, max_tumor_size=10, initial_wt=5, initial_mut=5)
    assert np.sum(env.state) == 1
    env.reset()
    assert np.sum(env.state) == 1

    env = LvEnv(normalize=False, max_tumor_size=10, initial_wt=5, initial_mut=5)
    assert np.sum(env.state) == 10
    env.reset()
    assert np.sum(env.state) == 10




