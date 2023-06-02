from physilearning.envs import LvEnv
from physilearning.callbacks import CopyConfigCallback, SaveOnBestTrainingRewardCallback
from physilearning.train import Trainer, make_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import pytest

def test_copy_config_callback():
    """
    Test the CopyConfigCallback
    """
    # create a LvEnv
    command = f'cp ./tests/test_cfg.yaml ./test_cfg.yaml'
    os.system(command)
    env = DummyVecEnv([make_env(LvEnv, config_file='test_cfg.yaml')])
    model = PPO('MlpPolicy', env, verbose=1)
    # train the model
    model.learn(total_timesteps=100, callback=CopyConfigCallback(config_file='test_cfg.yaml', logname='test_cfg'))
    # check if the config file is copied to the Training/Configs folder
    assert os.path.exists('./Training/Configs/test_cfg.yaml')
    # remove the copied config file
    #os.system('rm ./Training/Configs/test_cfg.yaml')
    # remove the logs
    os.system('rm -r ./Training/Logs/test*')

def test_save_on_best_training_reward_callback():
    """
    Test the SaveOnBestTrainingRewardCallback
    """
    # create a LvEnv
    command = f'cp ./tests/test_cfg.yaml ./test_cfg.yaml'
    os.system(command)
    env = DummyVecEnv([make_env(LvEnv, config_file='test_cfg.yaml')])
    model = PPO('MlpPolicy', env, verbose=1)
    # train the model
    model.learn(total_timesteps=100,
                callback=SaveOnBestTrainingRewardCallback(
                    check_freq=1, log_dir='./Training/Logs', save_dir='./Training/SavedModels', save_name='test'))
    # check if the model is saved to the Training/Models folder
    assert os.path.exists('./Training/SavedModels/test_best_reward.zip')
    # remove the copied config file
    os.system('rm ./Training/Configs/test_cfg.yaml')
    # remove the logs
    os.system('rm -r ./Training/Logs/test*')
    # remove the model
    os.system('rm ./Training/Models/test_best_reward.zip')

