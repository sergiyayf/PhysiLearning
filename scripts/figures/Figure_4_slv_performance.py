import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np
ex = {'font.size': 6,
          'font.weight': 'normal',
          'pdf.fonttype': 42,
          'font.family': 'sans-serif',
          'font.sans-serif': 'Arial',
          'errorbar.capsize': 2,
          }
plt.rcParams.update(ex)
# ex = {'font.size': 8,
#           'font.weight': 'normal',
#           'font.family': 'sans-serif'}
# plt.rcParams.update(ex)
# mpl.rcParams['pdf.fonttype'] = 42  # to make text editable in pdf output
# mpl.rcParams['font.sans-serif'] = ['Arial']  # to make it Arial

def get_ttps(filename, timesteps=50):
    ttps = []
    for i in range(timesteps):
        df = pd.read_hdf(filename, key=f'run_{i}')
        # find the largest index with non-zero Type 0 and Type 1
        initial_size = df['Type 0'][0] + df['Type 1'][0]
        nz = df[((df['Type 1']) / initial_size >= 1.0)]
        tot_idx = df[((df['Type 0'] + df['Type 1']) / initial_size >= 2.0)].index
        if len(nz) > 0:
            # append index when type 0 + type 1 is larger than 1.5
            ttps.append(nz.index[0] / 4)
        elif len(tot_idx) > 0:
            ttps.append(tot_idx[0] / 4)
        else:
            ttps.append(len(df) / 4)

    return ttps
def figure_setup(fig, ax, save_figure = False):
    # set plot font and fontsizes

    b = plot(fig, ax)

    ax.set_ylabel('Time to progression')
    ax.set_xlabel('Environment')
    # Set x ticks 1 - Train, 2 - Target
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Train', 'Target'])
    ax.set_xlim(0.5, 2.5)
    # fig.set_constrained_layout_pads(w_pad=10 / 72, h_pad=10 / 72, hspace=2 / 72, wspace=2 / 72)
    if save_figure:
        fig.savefig(r'scripts/figures/plots/Figure_1_heterogeneity.pdf', transparent = True)

    plt.show()


def plot(fig, ax):
    #PC_files_list = [f'./Evaluations/train_6_physicell_evals/train_6_run_{i}.h5' for i in range(1,11)]
    PC_files_list = [f'./Evaluations/1402_pcs_evals/run_{i}.h5' for i in range(1, 11)]
    sim_type_2 = ['PC' for i in range(500)]
    agent_name_2 = [i for i in range(1,11) for j in range(50) ]

    #LV_files_list = [f'Evaluations/train_6_lv_evals/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_{i}.h5' for i in range(1,11)]
    LV_files_list = [f'./Evaluations/2002_pc_evals_of_slvs/run_{i}.h5' for
                     i in range(1, 11)]
    sim_type = ['LV' for i in range(500)]
    agent_name = [i for i in range(1,11) for j in range(50) ]

    ttps = []
    for file in LV_files_list:
        tp = get_ttps(file)
        for t in tp:
            ttps.append(t)
    for file in PC_files_list:
        tp = get_ttps(file)
        for t in tp:
            ttps.append(t)


    sim_type.extend(sim_type_2)
    agent_name.extend(agent_name_2)

    data = {'ttp': ttps, 'sim_type': sim_type, 'agent': agent_name}
    df = pd.DataFrame(data)
    #average over sim type and agent
    df_grouped = df.groupby(['sim_type', 'agent'])
    mean = df_grouped['ttp'].mean()
    std = df_grouped['ttp'].std()

    # errorbar with x distances like jitter
    x0 = 1
    for i in range(1, 11):
        x = x0 + np.random.normal(0., 0.1)
        ax.errorbar(x, mean['LV'][i], yerr=std['LV'][i], fmt='o', label=f'LV {i}', color='grey', markersize=4.0)

    x0 = 2
    for i in range(1, 11):
        x = x0 + np.random.normal(0., 0.1)
        ax.errorbar(x, mean['PC'][i], yerr=std['PC'][i], fmt='o', label=f'PC {i}', color='darkgrey', markersize=4.0)


    return

if __name__ == '__main__':
    # setup pwd
    #os.chdir('/Users/saif/Desktop/Serhii/Projects/PhysiLearning')
    os.chdir('/home/saif/Projects/PhysiLearning')
    fig, ax = plt.subplots(figsize=(200 / 72, 150 / 72), constrained_layout=True)
    figure_setup(fig, ax, save_figure = False)