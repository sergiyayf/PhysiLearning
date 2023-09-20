import importlib
from unittest import mock
import pytest
from physilearning.envs import PcEnv
import yaml
import numpy as np
import os
from physilearning.tools.xml_reader import CfgRead
import subprocess


def test_env_creation():
    """
    Test environment creation.

    :param env_type: Environment type
    """

    EnvClass = getattr(importlib.import_module('physilearning.envs'), 'PcEnv')
    env = EnvClass()

    assert env is not None

def test_env_reset():
    """
    Test environment reset.

    :param env_type: Environment type
    """

    EnvClass = getattr(importlib.import_module('physilearning.envs'), 'PcEnv')
    env = EnvClass()

    # mock the _send_message and _start_slurm_physicell_job_step method to avoid sending messages to the server
    with (mock.patch.object(env, '_send_message') as mock_send_message,
            mock.patch.object(env, '_start_slurm_physicell_job_step') as mock_start_slurm_physicell_job_step,
            mock.patch.object(env, '_receive_message') as mock_receive_message):
            mock_receive_message.return_value = 'Type 0:12 t0_x: -200.0, 200.0, t0_y: -200.0, 200.0, Type 1:22 t1_x: -200.0, 200.0, t1_y: 200.0, -200.0, '
            obs = env.reset()


    assert obs is not None
    mock_send_message.assert_called_with('Start simulation')
    mock_start_slurm_physicell_job_step.assert_called_with()
    mock_receive_message.assert_called_with()


def test_normalization():
    env = PcEnv(observation_type='number', normalize=True, normalize_to=100, initial_wt=1, initial_mut=1, max_tumor_size=2)
    with (mock.patch.object(env, '_send_message') as mock_send_message,
            mock.patch.object(env, '_start_slurm_physicell_job_step') as mock_start_slurm_physicell_job_step,
            mock.patch.object(env, '_receive_message') as mock_receive_message):
        mock_receive_message.return_value = 'Type 0:1 t0_x: 0.0, t0_y: 0.0 Type 1:1 t1_x: 10.0, t1_y: 10.0'
        obs = env.reset()
    assert env.state[0] == 50.0
    assert obs[0] == 100.0


def test_get_cell_number():
    env = PcEnv()
    message = 'Type 0:12, t_x_pos = 0.0, Type 1:22, t_y_pos= 0.0'
    cell_number = env._get_cell_number(message)
    assert cell_number == (12, 22)


def test_get_image_obs():
    env = PcEnv(observation_type='image', image_size=11)
    message = 'Type 0:12 t0_x: -200.0, 200.0, t0_y: -200.0, 200.0, Type 1:22 t1_x: -200.0, 200.0, t1_y: 200.0, -200.0, '
    image = env._get_image_obs(message, action=0)
    assert np.sum(image) > 500


def test_get_tumor_volume_from_image():
    env = PcEnv(observation_type='image', image_size=11)
    message = 'Type 0:12 t0_x: -200.0, 200.0, t0_y: -200.0, 200.0, Type 1:22 t1_x: -200.0, 200.0, t1_y: 200.0, -200.0, '
    image = env._get_image_obs(message, action=0)
    num_cells = env._get_tumor_volume_from_image(image)
    assert num_cells == (2, 2)


def test_check_done():
    env = PcEnv(observation_type='number', normalize=0, max_tumor_size=2)
    env.state = [0, 0, 0]
    done = env._check_done(burden_type='number', message="Type 0:0, Type 1:0,")
    assert done == False

    env.state = [0, 3, 0]
    done = env._check_done(burden_type='number', message="Type 0:3, Type 1:2,")
    assert done == True


@pytest.mark.skipif(not os.path.exists('./simulations/PhysiCell_0'), reason='PhysiCell runnnig simulation directory does not exist')
def test_sample_patients():
    """
    Test environment reset.

    :param env_type: Environment type
    """
    np.random.seed(0)
    # os.chdir('/home/saif/Projects/PhysiLearning')
    config_file = './tests/test_cfg.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['env']['patient_sampling']['enable'] = True
    config['env']['patient_sampling']['patient_id'] = [80, 55]
    EnvClass = getattr(importlib.import_module('physilearning.envs'), 'PcEnv')
    env = EnvClass(config=config, patient_id=[80, 55])
    patients_list = []

    # mock the _send_message and _start_slurm_physicell_job_step method to avoid sending messages to the server
    with (mock.patch.object(env, '_send_message') as mock_send_message,
            mock.patch.object(env, '_start_slurm_physicell_job_step') as mock_start_slurm_physicell_job_step,
            mock.patch.object(env, '_receive_message') as mock_receive_message):
            mock_receive_message.return_value = 'Type 0:12 t0_x: -200.0, 200.0, t0_y: -200.0, 200.0, Type 1:22 t1_x: -200.0, 200.0, t1_y: 200.0, -200.0, '
            for i in range(10):
                env.reset()
                cfg_chkpt_file = env.config['patients'][env.patient_id]['PcEnv']['filename_chkpt']['value']
                xml_reader = CfgRead('./simulations/PhysiCell_0/config/PhysiCell_settings.xml')
                real_value = xml_reader.read_value(parent_nodes=['user_parameters'], parameter='filename_chkpt')
                assert cfg_chkpt_file == real_value
                patients_list.append(env.patient_id)


    assert len(set(patients_list)) == 2

