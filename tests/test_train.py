from physilearning import train
import yaml
import pytest
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack, DummyVecEnv, SubprocVecEnv
from physilearning.callbacks import CopyConfigCallback, SaveOnBestTrainingRewardCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO


@pytest.mark.parametrize("env_type", ["PcEnv", "LvEnv", "GridEnv"])
def test_setup_env(env_type):
    trainer = train.Trainer(config_file='./tests/test_cfg.yaml')
    assert trainer.env is None
    trainer.setup_env()
    assert trainer.env is not None
    with open('./tests/test_cfg.yaml', 'r') as f:
        yaml.load(f, Loader=yaml.FullLoader)
    trainer.env_type = env_type
    trainer.setup_env()
    assert trainer.env is not None


def test_1_env_vecframestack():
    trainer = train.Trainer(config_file='./tests/test_cfg.yaml')
    trainer.n_envs = 1
    trainer.env_type = "LvEnv"
    trainer.wrap = True
    trainer.wrapper = "VecFrameStack"
    trainer.setup_env()

    # assert isinstance(trainer.env, VecMonitor)
    assert isinstance(trainer.env, VecFrameStack)


def test_1_env_dummy():
    trainer = train.Trainer(config_file='./tests/test_cfg.yaml')
    trainer.n_envs = 1
    trainer.env_type = "LvEnv"
    trainer.wrap = True
    trainer.wrapper = "DummyVecEnv"
    trainer.setup_env()

    # assert isinstance(trainer.env, VecMonitor)
    assert isinstance(trainer.env, DummyVecEnv)


def test_not_defined_wrapper():
    trainer = train.Trainer(config_file='./tests/test_cfg.yaml')
    trainer.n_envs = 1
    trainer.env_type = "LvEnv"
    trainer.wrap = True
    trainer.wrapper = "NotDefinedWrapper"
    with pytest.raises(ValueError):
        trainer.setup_env()


@pytest.mark.parametrize("wrapper", ["DummyVecEnv", "VecFrameStack"])
def test_vector_envs(wrapper):
    trainer = train.Trainer(config_file='./tests/test_cfg.yaml')
    trainer.env_type = "LvEnv"
    trainer.wrap = True
    trainer.wrapper = wrapper
    trainer.n_envs = 2
    trainer.setup_env()
    assert trainer.env is not None


def test_subprocess_vec_env():
    trainer = train.Trainer(config_file='./tests/test_cfg.yaml')
    trainer.env_type = "LvEnv"
    trainer.wrap = True
    trainer.wrapper = "SubprocVecEnv"
    trainer.n_envs = 2
    trainer.setup_env()
    assert isinstance(trainer.env.unwrapped, SubprocVecEnv)


def test_setup_callbacks():
    trainer = train.Trainer(config_file='./tests/test_cfg.yaml')
    callbacks = trainer.setup_callbacks()
    trainer.env_type = "LvEnv"
    assert callbacks is not None
    assert isinstance(callbacks[1], CopyConfigCallback)
    trainer.save_freq = 0
    callbacks = trainer.setup_callbacks()
    assert isinstance(callbacks[0], CheckpointCallback)
    trainer.save_freq = 'best'
    callbacks = trainer.setup_callbacks()
    assert isinstance(callbacks[0], SaveOnBestTrainingRewardCallback)


def test_setup_model():
    trainer = train.Trainer(config_file='./tests/test_cfg.yaml')
    with pytest.raises(ValueError):
        trainer.setup_model()

    trainer.env_type = "GridEnv"
    trainer.setup_env()
    trainer.setup_model()
    assert trainer.model is not None
    assert trainer.model.policy is not None


def test_setup_non_existing_model():
    trainer = train.Trainer(config_file='./tests/test_cfg.yaml')
    trainer.env_type = "GridEnv"
    trainer.setup_env()
    trainer.model_name = "NonExistingModel"
    with pytest.raises(AttributeError):
        trainer.setup_model()


def test_setup_PPO():
    trainer = train.Trainer(config_file='./tests/test_cfg.yaml')
    trainer.env_type = "GridEnv"
    trainer.setup_env()
    trainer.model_name = "PPO"
    trainer.setup_model()
    assert isinstance(trainer.model, PPO)
    assert trainer.model.policy is not None