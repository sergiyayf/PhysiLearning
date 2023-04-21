# evaluation of the LatticeBased run
import matplotlib as mpl
mpl.use('TkAgg')
from physilearning.envs.grid_env import GridEnv
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def grid_env_sand():
    # setup environment
    env = GridEnv.from_yaml(r'.\Training\Configs\LatticeBased_23_03_CNN_1_env_not_stacked.yaml')

    # load pretrained model
    model = PPO.load(r'.\data\2403_grid_mela_models\LatticeBased_24_03_CNN_1_env_not_stacked_mela_final', env=env)

    done = False
    obs = env.reset()

    rew = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rew+=reward

    print('reward: ', rew)
    env.render()
    plt.show()


if __name__ == '__main__':
    filename = 'Evaluations/0_PcEnvEval0418_stacked_image_PC_test.csv_numpy.npy'
    trajectory = np.load(filename)
    fig, ax = plt.subplots()
    ims = []
    for i in range(10):
        im = ax.imshow(trajectory[:, :, i], animated=True, cmap='viridis', vmin=0, vmax=255)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    plt.show()
