from physilearning.envs import LvEnv
import pytest
import numpy as np
import os
import yaml


def test_observation_space():
    env = LvEnv(observation_type='number')
    assert env.observation_type == 'number'

    env = LvEnv(observation_type='image')
    assert env.observation_type == 'image'


def test_random_cell_number():
    env = LvEnv(initial_wt=5, normalize=False)
    assert env.initial_wt == 5
    env.reset()
    assert env.trajectory[0,0] >= 5

    initials_wt = []
    initials_mut = []
    for i in range(10):
        env = LvEnv(initial_wt='20-100', initial_mut='5pm1')
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
        env = LvEnv(initial_wt='20-100', initial_mut='5pm1')
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
    assert np.sum(env.state) >= 1

    env = LvEnv(normalize=False, max_tumor_size=10, initial_wt=5, initial_mut=5)
    assert np.sum(env.state) == 10
    env.reset()
    assert np.sum(env.state) >= 10


def test_get_image():
    env = LvEnv(observation_type='image', initial_wt=6500, initial_mut=0, max_tumor_size=7000,
                normalize=False, env_specific_params={'carrying_capacity': 6500}, treatment_time_step=1)
    env.reset()
    image = env._get_image(0)
    assert np.sum(image) >= 84*84*128

    env = LvEnv(observation_type='image', initial_wt=3000, initial_mut=0, max_tumor_size=7000,
                normalize=False, env_specific_params={'carrying_capacity': 6000}, treatment_time_step=1)
    env.reset()
    image = env._get_image(0)
    assert np.sum(image) >= 84 * 84 * 128/2

    env = LvEnv(observation_type='image', initial_wt=0, initial_mut=3000, max_tumor_size=7000,
                normalize=False, env_specific_params={'carrying_capacity': 6000}, treatment_time_step=1)
    env.reset()
    image = env._get_image(0)
    assert np.sum(image) >= 84 * 84 * env.mut_color / 2
    assert not (env.wt_color in image)
    assert (env.mut_color in image)


def test_delayed_growth_with_noise():
    np.random.seed(0)
    env = LvEnv(normalize=True, normalize_to=1000, initial_wt=1400)
    env.reset()
    pop_size = env.grow(1,0,'delayed')
    env.reset()
    pop_size_2 = env.grow(1, 0, 'delayed')
    env.reset()
    pop_size_with_noise = env.grow(1,0,'delayed_with_noise')
    assert pop_size_with_noise != pop_size and pop_size == pop_size_2
    pop_sizes_with_noise = []
    for i in range(1000):
        env.reset()
        pop_sizes_with_noise.append(env.grow(1,0,'delayed_with_noise'))
    assert pop_size-np.mean(pop_sizes_with_noise) < 1.e-2

def test_image_sampling():
    env = LvEnv(normalize=False, initial_wt=500, initial_mut=0, image_size=124)
    env.capacity = 500
    env.image_sampling_type = 'random'
    image = env._get_image(0)
    # all are wt_color
    assert np.sum(image) == 124*124*env.wt_color
    env.capacity = 1000
    image = env._get_image(0)
    first_rows_sum = np.sum(image[0,0:4,:])
    assert np.sum(image) == 124 * 124 * env.wt_color/2
    assert first_rows_sum >0

    env.image_sampling_type = 'dense'
    dense_image = env._get_image(0)
    first_rows_sum = np.sum(dense_image[0, 0:4, :])
    assert first_rows_sum == 0
