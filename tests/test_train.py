from physilearning import train
import yaml
import pytest


@pytest.mark.parametrize("env_type", ["PcEnv", "LvEnv", "GridEnv"])
def test_setup_env(env_type):
    trainer = train.Trainer(config_file='test_cfg.yaml')
    assert trainer.env is None
    trainer.setup_env()
    assert trainer.env is not None
    with open('test_cfg.yaml', 'r') as f:
        yaml.load(f, Loader=yaml.FullLoader)
    trainer.env_type = env_type
    trainer.setup_env()
    assert trainer.env is not None


@pytest.mark.parametrize("wrapper", ["DummyVecEnv", "VecFrameStack"])
def test_single_env_dummy_wrappers(wrapper):
    trainer = train.Trainer(config_file='test_cfg.yaml')
    trainer.wrap = True
    trainer.wrapper = wrapper
    trainer.setup_env()
    assert trainer.env is not None


@pytest.mark.parametrize("wrapper", ["DummyVecEnv", "VecFrameStack"])
def test_vector_envs(wrapper):
    trainer = train.Trainer(config_file='test_cfg.yaml')
    trainer.wrap = True
    trainer.wrapper = wrapper
    trainer.n_envs = 2
    trainer.setup_env()
    assert trainer.env is not None