from physilearning.envs import LvEnv
from stable_baselines3.common.monitor import Monitor
from physilearning.callbacks import CopyConfigCallback, SaveOnBestTrainingRewardCallback
from physilearning.train import Trainer, make_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import pytest

@pytest.mark.skip(reason='Not implemented yet')
def test_copy_config_callback():
    """
    Test the CopyConfigCallback
    """
    # create a LvEnv
    env = DummyVecEnv([make_env(LvEnv, config_file='test_cfg.yaml')])
    model = PPO('MlpPolicy', env, verbose=1)
    # train the model
    model.learn(total_timesteps=100, callback=CopyConfigCallback(config_file='test_cfg.yaml', logname='test'))
    # check if the config file is copied to the Training/Configs folder
    assert os.path.exists('./Training/Configs/test_cfg.yaml')
    # remove the copied config file
    os.system('rm ./Training/Configs/test_cfg.yaml')
    # remove the logs
    os.system('rm -r ./Training/Logs/test*')


