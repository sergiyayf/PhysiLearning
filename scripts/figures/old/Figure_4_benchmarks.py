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

def get_ttps(filename, timesteps=10):
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
    ex = {'font.size': 6,
          'font.weight': 'normal',
          'font.family': 'sans-serif'}
    plt.rcParams.update(ex)
    mpl.rcParams['pdf.fonttype'] = 42  # to make text editable in pdf output
    mpl.rcParams['font.sans-serif'] = ['Arial']  # to make it Arial

    sns.set(font_scale=0.6)
    # set plot size
    b = plot(fig, ax)
    print(ax.get_xticklabels())
    ax.legend()
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)
    # ax.set_yticklabels(ax.get_yticks(), fontsize=6)
    ax.set_ylabel('Time to progression', fontsize=6)
    fig.set_constrained_layout_pads(w_pad=10 / 72, h_pad=10 / 72, hspace=2 / 72, wspace=2 / 72)
    if save_figure:
        fig.savefig(r'one_comp_fig1_benchmark.svg', transparent = True)

    plt.show()


def plot(fig, ax):
    runs = 10
    PC_files_list = [f'./Evaluations/1402_pcs_evals/run_1_{i}.h5' for i in range(1,11)]
    #PC_files_list = [f'Evaluations/train_6_lv_evals/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_{i}.h5' for
    #                 i in range(1, 11)]
    sim_type_2 = ['LV-PC' for i in range(10*runs)]
    agent_name_2 = [i for i in range(1,11) for j in range(runs) ]

    LV_files_list = [
        f'./Evaluations/1402_lvs_evals/LvEnvEval__20250206_lv_1_{i}.h5'
        for i in range(1, 11)]
    #LV_files_list = [f'./Evaluations/2901_lv_long_buffer_2/LvEnvEval__20250129_lv_long_buffer_{i}.h5' for
    #                 i in range(1, 11)]
    sim_type = ['LV' for i in range(10 * runs)]
    agent_name = [i for i in range(1, 11) for j in range(runs)]

    #SLV_files_list = [f'Evaluations/train_6_on_slvenv/SLvEnvEval__slv_197_number_noise_on_treat20250109_2DLV_average_less_1_onehalf_day_{i}.h5' for i in range(1,11)]
    SLV_files_list = [
        f'./Evaluations/1402_lvs_evals_2/SLvEnvEval__only_rgr_mv_20250206_lv_1_{i}.h5'
        for i in range(1, 11)]
    sim_type_3 = ['SLV' for i in range(10*runs)]
    agent_name_3 = [i for i in range(1,11) for j in range(runs) ]

    # SLV trainings
    #LV_files_list = [f'Evaluations/slvenv_1901_train_average_2m/SLvEnvEval__agnt_20250116_SLV_train_{i}.h5' for
    #                i in range(1, 11)]
    #sim_type = ['train' for i in range(100)]
    #agent_name = [i for i in range(1, 11) for j in range(10)]
    # slv trained evals on pc
    #SLV_files_list = [f'./Evaluations/slv_trained_inter_evals/slv_trained_pc_eval_{i}.h5' for i in range(1,11)]
    #sim_type_3 = ['cross' for i in range(100)]
    #agent_name_3 = [i for i in range(1,11) for j in range(10) ]
    ttps = []
    for file in LV_files_list:
        tp = get_ttps(file)
        for t in tp:
            ttps.append(t)
    for file in PC_files_list:
        tp = get_ttps(file)
        for t in tp:
            ttps.append(t)
    for file in SLV_files_list:
        tp = get_ttps(file)
        for t in tp:
            ttps.append(t)


    sim_type.extend(sim_type_2)
    agent_name.extend(agent_name_2)
    sim_type.extend(sim_type_3)
    agent_name.extend(agent_name_3)

    data = {'ttp': ttps, 'sim_type': sim_type, 'agent': agent_name}
    df = pd.DataFrame(data)
    # Create the faceted boxplot
    g = sns.FacetGrid(df, col="agent", col_wrap=5, height=4, sharex=True, sharey=True)
    g.map_dataframe(sns.boxplot, x='sim_type', y='ttp', palette='coolwarm', width=0.5)
    g.map_dataframe(sns.stripplot, x='sim_type', y='ttp', color='black', jitter=True, size=3)
    #
    # g = sns.catplot(
    #     data=df,
    #     x='sim_type',
    #     y='ttp',
    #     hue='sim_type',
    #     col='agent',
    #     kind='strip',
    #     palette='coolwarm',
    #     col_order=[1,2,3,4,5,6,7,8,9,10],
    #
    # )

    # put g in the same figure
    g.fig.subplots_adjust(top=0.9)
    # adjust size of the figure
    g.fig.set_figwidth(10)
    g.fig.set_figheight(5)

    #g.fig.savefig('plots/boxplot_agents_degeneraccy.pdf', transparent = False)
    # Customize the layout


    return

if __name__ == '__main__':
    # setup pwd
    # os.chdir('/Users/saif/Desktop/Serhii/Projects/PhysiLearning')
    os.chdir('/')
    fig, ax = plt.subplots(figsize=(290 / 72, 160 / 72), constrained_layout=True)
    figure_setup(fig, ax, save_figure = False)