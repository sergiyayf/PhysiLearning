from physilearning.train import Trainer
from stable_baselines3 import PPO
import numpy as np

if __name__ == '__main__':
    config_file = 'config.yaml'
    trainer = Trainer(config_file)
    trainer.env_type = 'MeltdEnv'
    trainer.setup_env()
    model = PPO.load('./Training/SavedModels/2305_2d_meltd_cobra_r0_t17_best_reward.zip')

    obs = trainer.env.reset()
    obs = np.array([[0., 0., 0., 1000.]])

    action, _ = model.predict(obs)

    if action:
        print('Treat')
    else:
        print('Dont treat')
    obs, reward, done, info = trainer.env.step(action)
