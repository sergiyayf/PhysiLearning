import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
ex = {'font.size': 6,
          'font.weight': 'normal',
          'pdf.fonttype': 42,
          'font.family': 'sans-serif',
          'font.sans-serif': 'Arial',
          'errorbar.capsize': 2,
          }
plt.rcParams.update(ex)
def color_plot(df, max_time, ax=None, colormap = plt.cm.viridis):
    if ax is None:
        fig, ax = plt.subplots()
    # plot the number of resistat cells vs total, color the libe with index of the timepoint
    tot = df['Type 0'] + df['Type 1']
    res = df['Type 1']
    res = res[tot > 0]
    sus = df['Type 0']
    sus = sus[tot > 0]
    tot = tot[tot > 0]

    for i in range(max_time):
        ax.plot((res[i:i+2]), tot[i:i+2], color=colormap(i/max_time))
    ax.axhline(y=np.mean(tot), color='k', linestyle='--')
    ax.set_xscale('log')
    ax.set_ylim(0, 2)
    ax.set_xlim(0, 2)
    # plot colormap
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=max_time/4))
    sm.set_array([])
    #plt.colorbar(sm, ax=ax, label='Time')
    ax.set_xlabel('Resistant cells')
    ax.set_ylabel('Total cells')
    return ax

if __name__ == '__main__':
    os.chdir('/home/saif/Projects/PhysiLearning')

    #
    fig, ax = plt.subplots(figsize=(150 / 72, 80 / 72), constrained_layout=True)
    i=9
    j=2
    df = pd.read_hdf(f'./Evaluations/1402_lvs_evals/LvEnvEval__20250206_lv_1_{i}.h5', key=f'run_{j}')
    #df = pd.read_hdf(f'./Evaluations/1402_pcs_evals/run_{i}.h5', key=f'run_{j}')

    colormap = plt.cm.Oranges_r
    color_plot(df, 720, ax, colormap)
    fig.savefig(f'./scripts/figures/plots/Figure_2_res_vs_tot_lv.pdf', transparent=True)

    fig, ax = plt.subplots(figsize=(150 / 72, 80 / 72), constrained_layout=True)
    i = 9
    j = 0
    # df = pd.read_hdf(f'./Evaluations/1402_lvs_evals/LvEnvEval__20250206_lv_1_{i}.h5', key=f'run_{j}')
    df = pd.read_hdf(f'./Evaluations/1402_pcs_evals/run_{i}.h5', key=f'run_{j}')

    colormap = plt.cm.Oranges_r
    color_plot(df, 720, ax, colormap)
    fig.savefig(f'./scripts/figures/plots/Figure_2_res_vs_tot_pc.pdf', transparent=True)

    plt.show()
