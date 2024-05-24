from physilearning.train import Trainer
from stable_baselines3 import PPO
import numpy as np

if __name__ == '__main__':
    config_file = 'config.yaml'
    trainer = Trainer(config_file)
    trainer.env_type = 'MeltdEnv'
    trainer.setup_env()
    model = PPO.load('./Training/SavedModels/2305_2d_meltd_noise_agent_t3_best_reward.zip')

    obs = np.array([[0., 0., 0., 0., 0., 0., 0., 1000.]])

    action = model.predict(obs)


