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
    env = SLvEnv(normalize=True, normalize_to=1, max_tumor_size=10, initial_wt=5, initial_mut=5)
    assert np.sum(env.state) == 1
    env.reset()
    assert np.sum(env.state) == 1

    env = SLvEnv(normalize=False, max_tumor_size=10, initial_wt=5, initial_mut=5)
    assert np.sum(env.state) == 10
    env.reset()
    assert np.sum(env.state) == 10



def test_random_cell_number():
    env = SLvEnv(initial_wt=5)
    assert env.initial_wt == 5
    env.reset()
    assert env.state[0] == 5

    initials_wt = []
    initials_mut = []
    for i in range(10):
        env = SLvEnv(initial_wt='random', initial_mut='random')
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
        env = SLvEnv(initial_wt='random', initial_mut='random')
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
    env = SLvEnv(initial_wt=5, initial_mut=5, death_rate_mut=10, death_rate_wt=10, max_tumor_size=10)
    pop = env.grow(0, 1, 'no_flag')
    assert pop == 0

def test_competition_function():
    env = SLvEnv()
    env.capacity = 100
    env.state = [10,1,0]
    res = env._competition_function(10, 5)
    assert res == 9.9
    res = env._competition_function(10, 100)
    assert 0.99

def test_move_mutant():
    env = SLvEnv()
    env.state = [10, 1, 0]
    mut_x = env.mutant_x
    mut_y = env.mutant_y
    env._move_mutant(0, 1)
    assert mut_x == env.mutant_x and mut_y == env.mutant_y
    env._move_mutant(12, 10)
    assert mut_x == env.mutant_x and mut_y == env.mutant_y
    env._move_mutant(8, 160)
    assert mut_x != env.mutant_x or mut_y != env.mutant_y


def test_image_sampling():
    env = SLvEnv(initial_wt=500, initial_mut=0, image_size=124)
    env.capacity = 1000
    image = env._get_image(0)
    first_rows_sum = np.sum(image[0, 0:4, :])
    assert first_rows_sum == 0


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

    env = SLvEnv(initial_wt=500, initial_mut=0, image_size=124, observation_type='image')
    obs, rew, trunc, term, inf = env.step(0)
    assert obs.all() == env.image.all()

    env = SLvEnv(initial_wt=500, initial_mut=0, image_size=124, observation_type='multiobs')
    obs, rew, trunc, term, inf = env.step(0)
    assert obs['img'].all() == env.image.all()
