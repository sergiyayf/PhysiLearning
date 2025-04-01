import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np
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


def plot(fig, ax):
    runs = 50
    PC_files_list = [f'./Evaluations/1402_pcs_evals/run_{i}.h5' for i in range(1,11)]
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
        f'./Evaluations/1402_lvs_evals_2/SLvEnvEval__20250206_lv_1_{i}.h5'
        for i in range(1, 11)]
    sim_type_3 = ['SLV' for i in range(10*runs)]
    agent_name_3 = [i for i in range(1,11) for j in range(runs) ]

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


    #g.fig.savefig('plots/boxplot_agents_degeneraccy.pdf', transparent = False)
    # Customize the layout


    # plot average and std for all the agents
    df_grouped = df.groupby(['sim_type', 'agent'])
    mean = df_grouped['ttp'].mean()
    std = df_grouped['ttp'].std()
    # barplot for the average for each agent and sim_type
    for i in range(1, 11):
        ax.bar(i, mean['LV'][i], yerr=std['LV'][i], label=f'LV {i}', color='grey', alpha=0.5, width=0.2)
        ax.bar(i+0.25, mean['LV-PC'][i], yerr=std['LV-PC'][i], label=f'LV-PC {i}', color='darkgrey', alpha=0.5, width=0.2)
        ax.bar(i+0.5, mean['SLV'][i], yerr=std['SLV'][i], label=f'SLV {i}', color='black', alpha=0.5, width=0.2)
    ax.set_ylabel('TTF')

    return

if __name__ == '__main__':
    # setup pwd
    # os.chdir('/Users/saif/Desktop/Serhii/Projects/PhysiLearning')
    os.chdir('/home/saif/Projects/PhysiLearning')
    fig, ax = plt.subplots(figsize=(500 / 72, 130 / 72), constrained_layout=True)
    plot(fig, ax)
    #fig.savefig(r'./scripts/figures/plots/Figure_4_b_agents_lv_slv_pc.pdf', transparent = True)
