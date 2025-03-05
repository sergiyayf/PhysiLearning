import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
ex = {'font.size': 6,
          'font.weight': 'normal',
          'font.family': 'sans-serif'}
plt.rcParams.update(ex)
mpl.rcParams['pdf.fonttype'] = 42  # to make text editable in pdf output
mpl.rcParams['font.sans-serif'] = ['Arial']  # to make it Arial

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
def figure_setup(save_figure = False):
    # set plot font and fontsizes
    ex = {'font.size': 6,
          'font.weight': 'normal',
          'font.family': 'sans-serif'}
    plt.rcParams.update(ex)
    mpl.rcParams['pdf.fonttype'] = 42  # to make text editable in pdf output
    mpl.rcParams['font.sans-serif'] = ['Arial']  # to make it Arial

    sns.set(font_scale=0.6)
    # set plot size
    fig, ax = plot()
    print(ax.get_xticklabels())
    ax.legend()
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)
    # ax.set_yticklabels(ax.get_yticks(), fontsize=6)
    ax.set_ylabel('Time to progression', fontsize=6)
    fig.set_constrained_layout_pads(w_pad=10 / 72, h_pad=10 / 72, hspace=2 / 72, wspace=2 / 72)
    if save_figure:
        fig.savefig(r'one_comp_fig1_benchmark.svg', transparent = True)

    plt.show()


def plot():
    runs = 50
    #PC_files_list = [f'./Evaluations/train_6_physicell_evals/train_6_run_{i}.h5' for i in range(1,6)]
    PC_files_list = [f'./Evaluations/24012025_slvenv_physicell_evals/run_{i}.h5' for i in [1,2,4,9,10]]
    #PC_files_list = [f'./Evaluations/train_6_on_slvenv/SLvEnvEval__slv_207_number_noise_20250109_2DLV_average_less_1_onehalf_day_{i}.h5' for i in range(1, 6)]
    sim_type_2 = ['Validate' for i in range(5*runs)]
    agent_name_2 = [i for i in range(1,6) for j in range(runs) ]

    # LV_files_list = [f'Evaluations/train_6_lv_evals/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_{i}.h5' for
    #                  i in range(1, 6)]
    LV_files_list = [f'Evaluations/23012025_slv_train/SLvEnvEval__agnt_20250123_slv_run_{i}.h5' for
                     i in [1,2,4,9,10]]
    # LV_files_list = [f'./Evaluations/train_6_physicell_evals/train_6_run_{i}.h5' for
    #                  i in range(1, 6)]
    sim_type = ['Train' for i in range(5 * runs)]
    agent_name = [i for i in range(1, 6) for j in range(runs)]

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
    # Create the faceted boxplot
    g = sns.FacetGrid(df, col="agent", col_wrap=5, height=4, sharex=True, sharey=True)
    g.map_dataframe(sns.boxplot, x='sim_type', y='ttp', palette='coolwarm', width=0.5)
    g.map_dataframe(sns.stripplot, x='sim_type', y='ttp', color='black', jitter=True, size=3)

    # put g in the same figure
    g.fig.subplots_adjust(top=0.9)
    # adjust size of the figure
    g.fig.set_figwidth(10)
    g.fig.set_figheight(3)

    g.fig.set_constrained_layout(True)
    g.fig.savefig('./plots/slv_5_agents_for_poster.pdf', transparent=False)
    return

if __name__ == '__main__':
    # setup pwd
    # os.chdir('/Users/saif/Desktop/Serhii/Projects/PhysiLearning')
    os.chdir('/')
    figure_setup(save_figure = False)