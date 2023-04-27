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

def animate(filename, fig, ax):
    trajectory = np.load(filename)

    ims = []
    for i in range(len(trajectory[0, 0, :])):
        im = ax.imshow(trajectory[:, :, i], animated=True, cmap='viridis', vmin=0, vmax=255)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)
    return ani

if __name__ == '__main__':
    fig, ax = plt.subplots()
    filename = 'Evaluations/0_GridEnvEval0427_check_grid_no_treat_image_trajectory.npy'
    ani = animate(filename, fig, ax)

    fig2, ax2 = plt.subplots()
    filename2 = 'Evaluations/0_PcEnvEval0427_check_pc_no_treat_image_trajectory.npy'
    ani2 = animate(filename2, fig2, ax2)

    plt.show()
