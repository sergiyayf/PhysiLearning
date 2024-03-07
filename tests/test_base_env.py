import os
from physilearning.envs import BaseEnv
from gymnasium.spaces import Box
import numpy as np
import yaml

def test_construct_base_env():
    env = BaseEnv()
    assert env


def test_observation_space():
    obs_space = 'number'
    env = BaseEnv(observation_type=obs_space, normalize=True, normalize_to=20, see_resistance=True,
                  initial_wt=10, initial_mut=2, max_tumor_size=1.5)
    box = Box(low=0, high=30, shape=(3,))
    assert (env.observation_space == box)
    env = BaseEnv(observation_type=obs_space, normalize=True, normalize_to=20, see_resistance=False,
                  initial_wt=10, initial_mut=2, max_tumor_size=2.0)
    box = Box(low=0, high=40, shape=(2,))
    assert (env.observation_space == box)
    obs_space = 'image'
    env = BaseEnv(observation_type=obs_space, normalize=True, normalize_to=20, image_size=124)
    box = Box(low=0, high=255, shape=(1, 124, 124), dtype=np.uint8)
    assert (env.observation_space == box)
    obs_space = 'multiobs'
    env = BaseEnv(observation_type=obs_space, normalize=True, normalize_to=20, image_size=124,
                  initial_wt=10, initial_mut=2, max_tumor_size=2.0)
    box1 = Box(low=0, high=40, shape=(3,))
    box2 = Box(low=0, high=255, shape=(1, 124, 124), dtype=np.uint8)
    assert (env.observation_space['vec']==box1)
    assert (env.observation_space['img']==box2)
    obs_space = 'other'
    try:
        env = BaseEnv(observation_type=obs_space, normalize=True, normalize_to=20, image_size=124)
    except NotImplementedError:
        assert True


def test_patient_id():
    config = {'env': {'patient_sampling': {'enable': False, 'type': 'random'}}}
    env = BaseEnv(config=config, patient_id=80)
    assert env.patient_id == 80
    assert len(env.patient_id_list) == 1
    env_list = BaseEnv(config=config, patient_id=[80, 55])
    assert isinstance(env_list.patient_id, np.int64)
    assert len(env_list.patient_id_list) == 2
    # assert raising value error if patient_id is not int or list
    try:
        env = BaseEnv(config=config, patient_id='80')
    except ValueError:
        assert True


def test_patient_sampling():
    np.random.seed(0)
    # os.chdir('/home/saif/Projects/PhysiLearning')
    config_file = './tests/test_cfg.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['env']['patient_sampling']['patient_id'] = [80, 55]
    config['env']['patient_sampling']['enable'] = True
    env = BaseEnv(config=config, patient_id=[80, 55])
    patient_ids = []
    for _ in range(100):
        env._choose_new_patient()
        patient_ids.append(env.patient_id)
    # assert that patients ids are not all the same
    assert len(set(patient_ids)) > 1


def test_range_sampling():
    np.random.seed(0)
    # os.chdir('/home/saif/Projects/PhysiLearning')
    config_file = './tests/test_cfg.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['env']['patient_sampling']['patient_id'] = [1, 101]
    config['env']['patient_sampling']['enable'] = True
    config['env']['patient_sampling']['type'] = 'range'
    env = BaseEnv(config=config, patient_id=[1, 101])
    patient_ids = []
    for _ in range(100):
        patient_ids.append(env.patient_id)
        env._choose_new_patient()
    # assert that patients ids are range between 1 and 101
    assert patient_ids == list(range(1, 101))


def test_truncate():
    env = BaseEnv(initial_wt=10, initial_mut=2, normalize=False, max_tumor_size=15)
    trunc = env.truncate()
    assert trunc == False
    env.time += env.max_time
    trunc = env.truncate()
    assert trunc


def test_measure_response():
    env = BaseEnv(initial_wt=10, initial_mut=2, normalize=False)
    resp = env.measure_response()
    assert resp == 0
    env.time = 1
    env.trajectory[0,env.time] = 10
    env.trajectory[1,env.time] = 3
    env.trajectory[2,env.time] = 1
    env.state = [10, 3, 1]
    resp = env.measure_response()
    assert resp != 0

def test_terminate():
    env = BaseEnv(initial_wt=8, initial_mut=2, normalize=False, max_tumor_size=1.1)
    term = env.terminate()
    assert term == False
    env.time = 1
    env.trajectory[0,env.time] = 10
    env.trajectory[1,env.time] = 3
    env.trajectory[2,env.time] = 1
    env.state = [10, 2, 1]
    term = env.terminate()
    assert term

def test_render():
    env = BaseEnv(initial_wt=10, initial_mut=2, normalize=False, observation_type='image')
    from matplotlib.animation import ArtistAnimation
    ani = env.render()
    assert isinstance(ani, ArtistAnimation)