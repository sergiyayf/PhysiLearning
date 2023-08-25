from physilearning.envs import LvEnv
import pytest
import numpy as np


def test_observation_space():
    env = LvEnv(observation_type='number')
    assert env.observation_type == 'number'

    env = LvEnv(observation_type='image')
    assert env.observation_type == 'image'


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


def test_get_image():
    env = LvEnv(observation_type='image', initial_wt=6500, initial_mut=0, max_tumor_size=7000,
                normalize=False, env_specific_params={'carrying_capacity': 6500}, treatment_time_step=1)
    env.reset()
    image = env._get_image(0)
    assert np.sum(image) == 84*84*128

    env = LvEnv(observation_type='image', initial_wt=3000, initial_mut=0, max_tumor_size=7000,
                normalize=False, env_specific_params={'carrying_capacity': 6000}, treatment_time_step=1)
    env.reset()
    image = env._get_image(0)
    assert np.sum(image) == 84 * 84 * 128/2

    env = LvEnv(observation_type='image', initial_wt=0, initial_mut=3000, max_tumor_size=7000,
                normalize=False, env_specific_params={'carrying_capacity': 6000}, treatment_time_step=1)
    env.reset()
    image = env._get_image(0)
    assert np.sum(image) == 84 * 84 * env.mut_color / 2
    assert not (env.wt_color in image)
    assert (env.mut_color in image)


def test_step_image_obs():
    env = LvEnv(observation_type='image', initial_wt=500, initial_mut=0, max_tumor_size=7000,
                normalize=False, env_specific_params={'carrying_capacity': 6500, 'growth_function_flag': 'delayed'},
                treatment_time_step=1, growth_rate_wt=0.2, treat_death_rate_wt=0.5)
    env.reset()
    obs, reward, done, info = env.step(0)
    a = np.sum(obs)
    env.step(1)
    env.step(1)
    obs, reward, done, info = env.step(1)
    b = np.sum(obs)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    obs, reward, done, info = env.step(0)
    c = np.sum(obs)
    assert (a > b)
    assert (c > b)

