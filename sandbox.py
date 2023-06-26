# evaluation of the LatticeBased run
import matplotlib as mpl
mpl.use('TkAgg')
from physilearning.envs.grid_env import GridEnv
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from physicell_tools import pyMCDS
import pandas as pd
from physicell_tools.get_perifery import front_cells
from physicell_tools.leastsquares import leastsq_circle, plot_data_circle
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

def main():
    fig, ax = plt.subplots()
    filename = 'Evaluations/older_evals/0_PcEnvEval0507_gridenv_1env_full_node_rew5_image_trajectory.npy'
    ani = animate(filename, fig, ax)

    ani.save('./data/gm_cnn_trained/movies/grid_deployed_on_pc_same_as_cells.mp4', writer='ffmpeg', fps=10)

    # fig2, ax2 = plt.subplots()
    # filename2 = 'Evaluations/0_PcEnvEval0427_check_pc_no_treat_image_trajectory.npy'
    # ani2 = animate(filename2, fig2, ax2)

    plt.show()

def physicell_h5_trying():
    hdf_file = 'test_file.h5'
    cells_0 = pd.read_hdf(hdf_file, key='cell_info_00')
    #
    time_frames = range(0, 50, 1)
    cell_counts = pd.DataFrame()
    for t in time_frames:
        #pymc = pyMCDS.pyMCDS('output000000{:02d}.xml'.format(t) ,'simulations/outputs/manual_AT_output')
        #cell_info = pymc.get_cell_df()
        cell_info = pd.read_hdf(hdf_file, key='cell_info_{:02d}'.format(t))
        #cell_info.to_hdf(hdf_file, key='cell_info_{:02d}'.format(t))
        cell_count_type0 = len(cell_info[cell_info["cell_type"]==0])
        cell_count_type1 = len(cell_info[cell_info["cell_type"]==1])
        # add counts to dataframe with time as index
        cell_counts = pd.concat([cell_counts,
                                 pd.DataFrame({'type0': cell_count_type0,
                                               'type1': cell_count_type1}, index=[t])])

    cell_counts.plot(y=['type0', 'type1'])
    plt.show()

if __name__ == '__main__':
    pymc = pyMCDS.pyMCDS('final.xml' ,'./data/raven_22_06_patient_sims/PhysiCell_67/output')
    cell_info_0 = pymc.get_cell_df()
    fig, ax = plt.subplots()
    ax.scatter(cell_info_0['position_x'], cell_info_0['position_y'], c=cell_info_0['cell_type'])
    fig.tight_layout()

    sims = range(0, 100, 1)
    distance_to_center = []
    distance_to_front =  []
    for sim in sims:
        pymc = pyMCDS.pyMCDS('final.xml' ,f'./data/raven_22_06_patient_sims/PhysiCell_{sim}/output')
        cell_info = pymc.get_cell_df()

        positions, types = front_cells(cell_info)
        xc, yc, R, residu = leastsq_circle(positions[:, 0], positions[:, 1])
        # get cells of type 1
        type_1_cells = cell_info[cell_info['cell_type']==1]
        # calculate distance to center
        type_1_cells['distance_to_center'] = np.sqrt((type_1_cells['position_x'])**2 + (type_1_cells['position_y'])**2)
        type_1_cells['distance_to_front'] = (np.sqrt((type_1_cells['position_x']-xc)**2 + (type_1_cells['position_y']-yc)**2)).values - R
        # get unique clones
        unique_clones = type_1_cells['clone_ID'].unique()
        for clone in unique_clones:
            single_clone = type_1_cells[type_1_cells['clone_ID']==clone]
            # get average distance to center
            mean_dist_to_center = single_clone['distance_to_center'].mean()
            # get average distance to front
            mean_dist_to_front = single_clone['distance_to_front'].mean()
            # add to list
            distance_to_center.append(mean_dist_to_center)
            distance_to_front.append(mean_dist_to_front)

    # plot histogram
    fig3, ax3 = plt.subplots()
    ax3.hist(distance_to_center, bins=10)

    positions, types = front_cells(cell_info)
    xc, yc, R, residu = leastsq_circle(positions[:, 0], positions[:, 1])
    fig4, ax4 = plt.subplots()
    sns.histplot(data=distance_to_front, ax=ax4, bins=20)
    ax4.set_xlabel('Distance to front')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of resistant clones after 14 days of growth')
    #plot_data_circle(positions[:, 0], positions[:, 1], xc, yc, R)

    type_1_cell_0 = cell_info_0[cell_info_0['cell_type']==1]
    type_1_cell_0['distance_to_center'] = np.sqrt((type_1_cell_0['position_x'])**2 + (type_1_cell_0['position_y'])**2)
    type_1_cell_0['distance_to_front'] = (np.sqrt((type_1_cell_0['position_x']-xc)**2 + (type_1_cell_0['position_y']-yc)**2)).values - R


    plt.show()