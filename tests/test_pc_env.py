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
    with mock.patch.object(EnvClass, '_start_slurm_physicell_job_step') as mock_start_slurm_physicell_job_step:
            env = EnvClass()

    assert env is not None

def test_env_reset():
    """
    Test environment reset.

    :param env_type: Environment type
    """

    EnvClass = getattr(importlib.import_module('physilearning.envs'), 'PcEnv')

    # mock the _send_message and _start_slurm_physicell_job_step method to avoid sending messages to the server
    with (mock.patch.object(EnvClass, '_send_message') as mock_send_message,
            mock.patch.object(EnvClass, '_start_slurm_physicell_job_step') as mock_start_slurm_physicell_job_step,
            mock.patch.object(EnvClass, '_receive_message') as mock_receive_message):
            mock_receive_message.return_value = 'Type 0:12 t0_x: -200.0, 200.0, t0_y: -200.0, 200.0, Type 1:22 t1_x: -200.0, 200.0, t1_y: 200.0, -200.0, '
            env = EnvClass(initial_wt=1, initial_mut=1, observation_type='number', normalize=False)
            obs, _ = env.reset()


    assert obs == [34, 0]
    mock_send_message.assert_called_with('Reset')
    mock_start_slurm_physicell_job_step.assert_called_with()
    mock_receive_message.assert_called_with()


def test_normalization():

    with (mock.patch.object(PcEnv, '_send_message') as mock_send_message,
            mock.patch.object(PcEnv, '_start_slurm_physicell_job_step') as mock_start_slurm_physicell_job_step,
            mock.patch.object(PcEnv, '_receive_message') as mock_receive_message):
        env = PcEnv(observation_type='number', normalize=True, normalize_to=100, initial_wt=1, initial_mut=1,
                    max_tumor_size=2, see_resistance=True)
        mock_receive_message.return_value = 'Type 0:1 t0_x: 0.0, t0_y: 0.0, Type 1:1 t1_x: 10.0, t1_y: 10.0, '
        obs, _ = env.reset()
    assert env.state[0] == 50.0
    assert obs[0]+obs[1] == 100.0


def test_get_cell_number():
    with mock.patch.object(PcEnv, '_start_slurm_physicell_job_step') as mock_start_slurm_physicell_job_step:
        env = PcEnv()
        message = 'Type 0:12, t_x_pos = 0.0, Type 1:22, t_y_pos= 0.0'
        cell_number = env._get_cell_number(message)
    assert cell_number == (12, 22)


def test_get_image_obs():
    with mock.patch.object(PcEnv, '_start_slurm_physicell_job_step') as mock_start_slurm_physicell_job_step:
        env = PcEnv(observation_type='image', image_size=11)
        message = 'Type 0:12 t0_x: -200.0, 200.0, t0_y: -200.0, 200.0, Type 1:22 t1_x: -200.0, 200.0, t1_y: 200.0, -200.0, '
        image = env._get_image_obs(message, action=0)
    assert np.sum(image) > 500


def test_get_tumor_volume_from_image():
    with mock.patch.object(PcEnv, '_start_slurm_physicell_job_step') as mock_start_slurm_physicell_job_step:
        env = PcEnv(observation_type='image', image_size=11)
        message = 'Type 0:12 t0_x: -200.0, 200.0, t0_y: -200.0, 200.0, Type 1:22 t1_x: -200.0, 200.0, t1_y: 200.0, -200.0, '
        image = env._get_image_obs(message, action=0)
        num_cells = env._get_tumor_volume_from_image(image)
    assert num_cells == (2, 2)

def test_get_df_from_message():
    with mock.patch.object(PcEnv, '_start_slurm_physicell_job_step') as mock_start_slurm_physicell_job_step:
        env = PcEnv(observation_type='number', normalize=False, max_tumor_size=2)
        message = ('Type 0:12 Type 1:22 t0_x: -200.0, 200.0, t0_y: -200.0, 200.0, t0_z: 0.0, 10.0,'
                   ' t1_x: -200.0, 200.0, t1_y: 200.0, -200.0, t1_z: 0.0, 10.0, ')
        df = env._get_df_from_message(message)
    print(df)
    assert True


def test_measure_radius():
    with mock.patch.object(PcEnv, '_start_slurm_physicell_job_step') as mock_start_slurm_physicell_job_step:
        env = PcEnv(observation_type='number', normalize=False, max_tumor_size=2)
        message = ('Type 0:12 Type 1:22 t0_x: -200.0, 200.0, t0_y: -200.0, 200.0, t0_z: 0.0, 0.0,'
                   ' t1_x: -200.0, 200.0, t1_y: 200.0, -200.0, t1_z: 0.0, 0.0, ')
        df = env._get_df_from_message(message)
        radius = env._measure_radius()
    assert radius-np.sqrt(200**2+200**2) < 1.e-3

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

    patients_list = []

    # mock the _send_message and _start_slurm_physicell_job_step method to avoid sending messages to the server
    with (mock.patch.object(PcEnv, '_send_message') as mock_send_message,
            mock.patch.object(PcEnv, '_start_slurm_physicell_job_step') as mock_start_slurm_physicell_job_step,
            mock.patch.object(PcEnv, '_receive_message') as mock_receive_message):
            mock_receive_message.return_value = 'Type 0:12 t0_x: -200.0, 200.0, t0_y: -200.0, 200.0, Type 1:22 t1_x: -200.0, 200.0, t1_y: 200.0, -200.0, '
            env = EnvClass(config=config, patient_id=[80, 55])
            for i in range(10):
                env.reset()
                cfg_chkpt_file = env.config['patients'][env.patient_id]['PcEnv']['filename_chkpt']['value']
                xml_reader = CfgRead('./simulations/PhysiCell_0/config/PhysiCell_settings.xml')
                real_value = xml_reader.read_value(parent_nodes=['user_parameters'], parameter='filename_chkpt')
                assert cfg_chkpt_file == real_value
                patients_list.append(env.patient_id)


    assert len(set(patients_list)) == 2

@pytest.mark.skipif(not os.path.exists('./simulations/PhysiCell_0'), reason='PhysiCell runnnig simulation directory does not exist')
def test_sample_patients_range():
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
    config['env']['patient_sampling']['patient_id'] = [1, 11]
    config['env']['patient_sampling']['type'] = 'range'
    EnvClass = getattr(importlib.import_module('physilearning.envs'), 'PcEnv')

    patients_list = []

    # mock the _send_message and _start_slurm_physicell_job_step method to avoid sending messages to the server
    with (mock.patch.object(EnvClass, '_send_message') as mock_send_message,
            mock.patch.object(EnvClass, '_start_slurm_physicell_job_step') as mock_start_slurm_physicell_job_step,
            mock.patch.object(EnvClass, '_receive_message') as mock_receive_message):
            mock_receive_message.return_value = 'Type 0:12 t0_x: -200.0, 200.0, t0_y: -200.0, 200.0, Type 1:22 t1_x: -200.0, 200.0, t1_y: 200.0, -200.0, '
            env = EnvClass(config=config, patient_id=[80, 55])
            for i in range(10):
                patients_list.append(env.patient_id)
                env.reset()
                cfg_chkpt_file = f'./../paper_presims/patient_{env.patient_id}/final'
                xml_reader = CfgRead('./simulations/PhysiCell_0/config/PhysiCell_settings.xml')
                real_value = xml_reader.read_value(parent_nodes=['user_parameters'], parameter='filename_chkpt')
                assert cfg_chkpt_file == real_value


    assert patients_list == list(range(1, 11))