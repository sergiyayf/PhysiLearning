import os
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
ex = {'font.size': 6,
          'font.weight': 'normal',
          'pdf.fonttype': 42,
          'font.family': 'sans-serif',
          'font.sans-serif': 'Arial',
          'errorbar.capsize': 2,
          }
plt.rcParams.update(ex)

def plot( df, title, scale='linear', truncate=False, ax=None, c='black'):
    if ax is None:
        fig, ax = plt.subplots()
    if truncate:
        initial_size = df['Type 0'][0] + df['Type 1'][0]
        truncated = df[((df['Type 0'] + df['Type 1'])/initial_size >= 1.4)]
        print(truncated)
        index = truncated.index[0]
        # replace df with zeros after index
        df.loc[index:, 'Type 0'] = 0
        df.loc[index:, 'Type 1'] = 0
        df.loc[index:, 'Treatment'] = 0

    skip = 1
    normalize = 2
    sus = np.array(df['Type 0'].values[::skip] )#/ (df['Type 0'][normalize] + df['Type 1'][normalize]))
    res = np.array(df['Type 1'].values[::skip] )#/ (df['Type 0'][normalize] + df['Type 1'][normalize]))
    # sus = np.array(df['Type 0'].values)*70
    # res = np.array(df['Type 1'].values)*70
    tot = sus + res
    time = df.index[::skip] / skip
    # only do for nonzero tot
    time = time[tot > 0]
    sus = sus[tot > 0]
    res = res[tot > 0]
    tot = tot[tot > 0]

    treatment_color = '#A8DADC'
    color = '#EBAA42'
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('# cells (norm)')
    ax.plot(time, res, color=color, marker='x', markersize=1, lw=1)
    ax.plot(time[::6], res[::6], color=color, marker='o',
            label='Resistant', markersize=2, lw=0)
    # ax.plot(data['Day'], data['Red'], color=color)
    ax.tick_params(axis='y')

    color = '#5E82B8'
    ax.plot(time, sus, color=color, marker='x',
            label='Sensitive', markersize=1, lw=1)
    ax.plot(time[::6], sus[::6], color=color, marker='o', markersize=2, lw=0)
    ax.plot(time, tot, color='k', marker='x', markersize=1, lw=1)
    ax.plot(time[::6], tot[::6], color='k', marker='o',
            label='Total', markersize=2, lw=0)

    ax.set_yscale(scale)
    ax.hlines(1.0, 0, time[-1], color='k', linestyle='--')
    treat = np.array(df['Treatment'].values[::skip])
    treat = treat[:len(time)]
    # replace 0s that are directly after 1 with 1s
    #treat = np.where(treat == 0, np.roll(treat, -1), treat)
    for t in range(len(time)-1):
        if treat[t] == 1:
            ax.axvspan((t-1), t, color=treatment_color)

    ax.set_xlabel('Time, Days')
    ax.set_ylabel('Relative tumor size')

    # additional find maxima and minima and plot lines
    y = np.array(df['Type 0']+df['Type 1'])[::skip]
    maxima = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    # add y[0]
    maxima = np.insert(maxima, 0, True)
    maxima_indices = np.where(maxima)[0]   # Adjust index due to slicing
    m = y[maxima_indices]
    minima = (y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])
    minima_indices = np.where(minima)[0] + 1  # Adjust index due to slicing
    n = y[minima_indices]
    # plot maxima
    ax.plot(maxima_indices, m, 'r--', lw =1)
    # plot minima
    ax.plot(minima_indices, n, 'b--', lw=1)
    # ax.legend()
    return ax

if __name__ == '__main__':

    os.chdir('/home/saif/Projects/PhysiLearning')

    fig, ax = plt.subplots(figsize=(200 / 72, 70 / 72), constrained_layout=True)
    # worst agnt 9 run 0 PC run 2 LV
    # best agnt 4 run 0 PC run 3 LV
    df = pd.read_hdf(f'./Evaluations/1402_pcs_evals/run_9.h5', key=f'run_0')
    # df = pd.read_hdf(f'./Evaluations/1402_lvs_evals/LvEnvEval__20250206_lv_1_4.h5', key=f'run_3')
    plot(df, f'LV', scale='linear', truncate=False, ax = ax, c='red')
    ax.set_xlim(0, 720)
    ax.set_ylim(0, 1.5)
    # divide x ticks by 4
    ticks = ax.get_xticks()
    tick_labels = [int(t/4) for t in ticks]
    ax.set_xticklabels(tick_labels)
    ax.set_yscale('linear')
    fig.savefig('./scripts/figures/plots/Figure_2_trajectory_PC_worst_agent.pdf', transparent = True)

    fig, ax = plt.subplots(figsize=(200 / 72, 70 / 72), constrained_layout=True)
    # worst agnt 9 run 0 PC run 2 LV
    # best agnt 4 run 0 PC run 3 LV
    df = pd.read_hdf(f'./Evaluations/1402_pcs_evals/run_4.h5', key=f'run_0')
    # df = pd.read_hdf(f'./Evaluations/1402_lvs_evals/LvEnvEval__20250206_lv_1_4.h5', key=f'run_3')
    plot(df, f'LV', scale='linear', truncate=False, ax=ax, c='red')
    ax.set_xlim(0, 720)
    ax.set_ylim(0, 1.5)
    # divide x ticks by 4
    ticks = ax.get_xticks()
    tick_labels = [int(t / 4) for t in ticks]
    ax.set_xticklabels(tick_labels)
    ax.set_yscale('linear')
    fig.savefig('./scripts/figures/plots/Figure_2_trajectory_PC_best_agent.pdf', transparent=True)

    fig, ax = plt.subplots(figsize=(200 / 72, 70 / 72), constrained_layout=True)
    # worst agnt 9 run 0 PC run 2 LV
    # best agnt 4 run 0 PC run 3 LV
    # df = pd.read_hdf(f'./Evaluations/1402_pcs_evals/run_9.h5', key=f'run_0')
    df = pd.read_hdf(f'./Evaluations/1402_lvs_evals/LvEnvEval__20250206_lv_1_9.h5', key=f'run_2')
    plot(df, f'LV', scale='linear', truncate=False, ax=ax, c='red')
    ax.set_xlim(0, 720)
    ax.set_ylim(0, 1.5)
    # divide x ticks by 4
    ticks = ax.get_xticks()
    tick_labels = [int(t / 4) for t in ticks]
    ax.set_xticklabels(tick_labels)
    ax.set_yscale('linear')
    fig.savefig('./scripts/figures/plots/Figure_2_trajectory_LV_worst_agent.pdf', transparent=True)

    fig, ax = plt.subplots(figsize=(200 / 72, 70 / 72), constrained_layout=True)
    # worst agnt 9 run 0 PC run 2 LV
    # best agnt 4 run 0 PC run 3 LV
    # df = pd.read_hdf(f'./Evaluations/1402_pcs_evals/run_9.h5', key=f'run_0')
    df = pd.read_hdf(f'./Evaluations/1402_lvs_evals/LvEnvEval__20250206_lv_1_4.h5', key=f'run_3')
    plot(df, f'LV', scale='linear', truncate=False, ax=ax, c='red')
    ax.set_xlim(0, 720)
    ax.set_ylim(0, 1.5)
    # divide x ticks by 4
    ticks = ax.get_xticks()
    tick_labels = [int(t / 4) for t in ticks]
    ax.set_xticklabels(tick_labels)
    ax.set_yscale('linear')
    fig.savefig('./scripts/figures/plots/Figure_2_trajectory_LV_best_agent.pdf', transparent=True)
    #
    plt.show()
