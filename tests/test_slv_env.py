from physilearning.envs import SLvEnv
import pytest
import numpy as np
import os
import yaml


def test_observation_space():
    env = SLvEnv(observation_type='number')
    assert env.observation_type == 'number'

    env = SLvEnv(observation_type='image')
    assert env.observation_type == 'image'

def test_normalization():

    config = yaml.safe_load(open('./tests/test_cfg_eval.yaml'))
    env = SLvEnv(normalize=True, normalize_to=1, max_tumor_size=10, initial_wt=5, initial_mut=5, config=config)
    assert np.sum(env.state) == 1
    env.reset()
    assert np.sum(env.state) == 1

    config = yaml.safe_load(open('./tests/test_cfg_eval.yaml'))
    env = SLvEnv(normalize=False, max_tumor_size=10, initial_wt=5, initial_mut=5, config=config)
    assert np.sum(env.state) == 10
    env.reset()
    assert np.sum(env.state) == 10



def test_random_cell_number():
    config = yaml.safe_load(open('./tests/test_cfg_eval.yaml'))
    env = SLvEnv(initial_wt=5, normalize=False, config=config)
    assert env.initial_wt == 5
    env.reset()
    assert env.state[0] == 5

    initials_wt = []
    initials_mut = []
    for i in range(10):
        env = SLvEnv(initial_wt='10-20', initial_mut='5pm2')
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
        config = yaml.safe_load(open('./tests/test_cfg_eval.yaml'))
        env = SLvEnv(initial_wt='10-20', initial_mut='5pm2', config=config)
        env.reset()
        initial_wt = env.initial_wt
        initial_mut = env.initial_mut
        initials_wt.append(initial_wt)
        initials_mut.append(initial_mut)

    diff_wt = [initials_wt[i] - initials_wt[i+1] for i in range(len(initials_wt)-1)]
    diff_mut = [initials_mut[i] - initials_mut[i+1] for i in range(len(initials_mut)-1)]
    assert np.sum(diff_wt) != 0
    assert np.sum(diff_mut) != 0

def test_die():
    env = SLvEnv(initial_wt=5, initial_mut=5, death_rate_mut=100, death_rate_wt=100, max_tumor_size=100000)
    pop = env.grow(0, 1, 'no_flag')
    assert pop == 0

def test_move_mutant():
    env = SLvEnv()
    env.state = [10, 1, 0]
    env.mutant_radial_position = 10
    env.radius = 100
    env._move_mutant(0, 1)
    assert env.mutant_radial_position == 100

    env.mutant_radial_position = 10
    env.radius = 100
    positions = []
    for i in range(10):
        env._move_mutant(10, 21)
        positions.append(env.mutant_radial_position)
    # some positions are different than 10 and 100
    assert np.diff(positions).any() != 0


def test_step():
    env = SLvEnv(initial_wt=500, initial_mut=0, image_size=124, observation_type='number')
    env.capacity = 1000
    env.see_resistance = False
    obs, rew, trunc, term, inf = env.step(0)
    assert obs[0] == np.sum(env.state[0:2])
    assert rew is not None
    assert trunc == False
    assert term == False
    assert inf == {}

