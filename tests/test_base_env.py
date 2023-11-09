import os
from physilearning.envs import BaseEnv
from gym.spaces import Box
import numpy as np
import yaml

def test_construct_base_env():
    config = {'env': {'patient_sampling': {'enable': False}}}
    env = BaseEnv(config=config)
    assert env


def test_observation_space():
    obs_space = 'number'
    config = {'env': {'patient_sampling': {'enable': False}}}
    env = BaseEnv(config=config, observation_type=obs_space, normalize=True, normalize_to=20)
    box = Box(low=0, high=20, shape=(3,))
    assert (env.observation_space == box)


def test_patient_id():
    config = {'env': {'patient_sampling': {'enable': False, 'type': 'random'}}}
    env = BaseEnv(config=config, patient_id=80)
    assert env.patient_id == 80
    assert len(env.patient_id_list) == 1
    env_list = BaseEnv(config=config, patient_id=[80, 55])
    assert isinstance(env_list.patient_id, np.int64)
    assert len(env_list.patient_id_list) == 2


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

