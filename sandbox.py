# evaluation of the LatticeBased run
import matplotlib as mpl
mpl.use('TkAgg')
from physilearning.envs.grid_env import GridEnv
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns


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
        arr = trajectory[:, :, i]
        # change x and y axis
        arr = np.flip(arr, 0)
        im = ax.imshow(arr, animated=True, cmap='viridis', vmin=0, vmax=255)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)
    return ani

def boxplots():
    eval_pcenv_0505_train_2_rewards = [441.98, 473.3, 436.14, 418.68, 405.69,
                                       408.44, 446.9, 395.89, 473.58, 481.8]
    eval_pcenv_0505_train_2_len = [477, 507, 469, 449, 436, 439, 482, 426, 511, 520]
    eval_pcenv_0505_raven_finding_rewards = [551.9, 530.6, 446.8, 543.8, 456.5,
                                             491.5, 475.9, 409.3, 491.5, 422.3]
    eval_pcenv_0505_raven_finding_len = [594, 572, 479, 588, 493,
                                         530, 529, 513, 443, 455]
    eval_grid_env_0507_len = [1000, 1000, 1000, 810, 956,
                              1000, 1000, 924, 1000, 793]
    eval_grid_env_0507_len_on_pc = [407, 450, 397, 435, 440,
                                    421, 393, 388, 485, 489]
    eval_pcenv_AT50 = [641, 450, 655, 481, 527,
                       563, 576, 497, 501, 445]
    eval_trained_on_pc_only_job = [490, 832, 500, 468, 468,
                                   500, 504, 623, 492, 550]
    eval_mtd = [328, 411, 360, 381, 343,
                368, 344, 348, 391, 343]

    fig, ax = plt.subplots()
    sns.boxplot(data=[eval_mtd, eval_grid_env_0507_len_on_pc, eval_pcenv_AT50, eval_pcenv_0505_raven_finding_len],
                ax=ax)
    # add legends
    ax.set_xticklabels(['MTD', 'Lattice-based', 'AT50', 'Agent-based'])
    ax.legend()
    fig.savefig('./boxplot_compare_different_strategies.pdf', transparent=True)

if __name__ == '__main__':
    fig, ax = plt.subplots()
    filename = 'Evaluations/0_PcEnvEval0507_gridenv_1env_full_node_rew5_image_trajectory.npy'
    ani = animate(filename, fig, ax)

    ani.save('./data/gm_cnn_trained/movies/grid_deployed_on_pc_same_as_cells.mp4', writer='ffmpeg', fps=10)

    # fig2, ax2 = plt.subplots()
    # filename2 = 'Evaluations/0_PcEnvEval0427_check_pc_no_treat_image_trajectory.npy'
    # ani2 = animate(filename2, fig2, ax2)

    plt.show()
