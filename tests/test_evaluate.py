import pytest
from physilearning import evaluate
import importlib
from physilearning.evaluate import DontWrapError
from stable_baselines3.common.vec_env import VecMonitor
from physilearning.train import make_env
from stable_baselines3.common.vec_env import DummyVecEnv
import os


@pytest.mark.parametrize('env_type', ['LvEnv'])
def test_at_fixed(env_type):
    """
    Test fixed adaptive therapy evaluation
    """
    EnvClass = getattr(importlib.import_module('physilearning.envs'), env_type)
    env = EnvClass(normalize_to=4, max_tumor_size=10, initial_wt=3, initial_mut=1)
    env.observation_type = 'number'
    env.reset()
    env.step(0)
    env.state[0] = 9.
    env.state[1] = .4

    treatment = evaluate.fixed_at(env, threshold=0.8, at_type='fixed')
    assert treatment == 1
    treatment = evaluate.fixed_at(env, threshold=1.95, at_type='fixed')
    assert treatment == 1
    threshold = 9.4/4-1.e-3
    treatment = evaluate.fixed_at(env, threshold=threshold, at_type='fixed')
    assert treatment == 1
    threshold = 9.4 / 4
    treatment = evaluate.fixed_at(env, threshold=threshold, at_type='fixed')
    assert treatment == 0
    treatment = evaluate.fixed_at(env, at_type='mtd')
    assert treatment == 1
    treatment = evaluate.fixed_at(env, at_type='no_treatment', threshold=0.95)
    assert treatment == 0


@pytest.mark.parametrize('env_type', ['LvEnv'])
def test_at_fixed_with_vector_monitor(env_type):
    """
    Test fixed adaptive therapy evaluation with vector monitor
    """
    EnvClass = getattr(importlib.import_module('physilearning.envs'), env_type)
    env = DummyVecEnv([make_env(EnvClass, config_file='./tests/test_cfg.yaml')])
    env = VecMonitor(env)
    with pytest.raises(DontWrapError):
        evaluate.fixed_at(env, threshold=0.8, at_type='fixed')


@pytest.mark.parametrize('env_type', ['LvEnv'])
def test_zhang_et_al(env_type):
    """
    Test Zhang et al adaptive therapy evaluation
    """
    EnvClass = getattr(importlib.import_module('physilearning.envs'), env_type)
    env = EnvClass(normalize_to=4, max_tumor_size=10, initial_wt=3, initial_mut=1, observation_type='number')
    env.observation_type = 'number'
    env.reset()
    env.step(0)
    env.state[0] = 9

    treatment = evaluate.fixed_at(env, at_type='zhang_et_al')
    assert treatment == 1
    env.state[0] = 1
    treatment = evaluate.fixed_at(env, at_type='zhang_et_al')
    assert treatment == 0
    env.step(1)
    env.state[0] = 3
    treatment = evaluate.fixed_at(env, at_type='zhang_et_al', threshold=0.5)
    assert treatment == 1


def test_evaluation_class_wrapper_handling_image_observation():
    from physilearning.envs import GridEnv
    env = DummyVecEnv([make_env(GridEnv, config_file='./tests/test_cfg.yaml')])
    env = VecMonitor(env)
    eval = evaluate.Evaluation(env, config_file='./tests/test_cfg.yaml')
    trajectory_dim = eval.image_trajectory.shape[0]
    assert trajectory_dim > 4

    env = GridEnv(observation_type='image')
    eval = evaluate.Evaluation(env, config_file='./tests/test_cfg.yaml')
    trajectory_dim = eval.image_trajectory.shape[0]
    assert trajectory_dim > 4


def test_evaluation_class_wrapper_handling_number_observation():
    from physilearning.envs import LvEnv
    env = DummyVecEnv([make_env(LvEnv, config_file='./tests/test_cfg.yaml')])
    env = VecMonitor(env)
    eval = evaluate.Evaluation(env, config_file='./tests/test_cfg.yaml')
    trajectory_dim = eval.trajectory.shape[0]
    assert trajectory_dim == 3

    env = LvEnv(observation_type='number')
    eval = evaluate.Evaluation(env, config_file='./tests/test_cfg.yaml')
    trajectory_dim = eval.trajectory.shape[0]
    assert trajectory_dim == 3


def test_evaluate_run():
    """
    Integration test for evaluation run
    """
    evaluate.evaluate('./tests/test_cfg_eval.yaml')
    assert os.path.exists('./Evaluations/LvEnvEval__test.h5')
    os.system('rm -r ./Evaluations/LvEnvEval__test.h5')

