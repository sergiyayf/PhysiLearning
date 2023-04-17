from physilearning import train
import yaml
import pytest

@pytest.mark.parametrize("env_type", ["PcEnv", "LvEnv", "GridEnv"])
def test_train(env_type):
    trainer = train.Trainer(config_file='test_cfg.yaml')
    assert trainer.env is None
    trainer.setup_env()
    assert trainer.env is not None
    with open('test_cfg.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    trainer.env_type = env_type
    trainer.setup_env()
    assert trainer.env.name == env_type