import os

import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

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
    ax.plot(time, res, color=color, marker='x',
                label='Resistant', markersize=4)
    # ax.plot(data['Day'], data['Red'], color=color)
    ax.tick_params(axis='y')

    color = '#5E82B8'
    ax.plot(time, sus, color=color, marker='x',
            label='Sensitive', markersize=4)

    # ax.legend()
    # ax.plot(time[::skip], sus[::skip], color='g')
    # ax.plot(time[::skip], res[::skip], color='r')
    # ax.plot(time[::skip], tot[::skip], color=c)

    # ax.legend()
    #ax.set_title(title)
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
    # ax.legend()
    return ax

##df = pd.read_hdf(f'./Evaluations/train_6_physicell_evals/train_6_run_{i}.h5', key=f'run_{j}')

os.chdir('/')
for i in range(0,15):
    fig, ax = plt.subplots(figsize=(4,2), constrained_layout=True)
    #df = pd.read_hdf(f'./Evaluations/train_6_physicell_evals/train_6_run_4.h5', key=f'run_1')
    df = pd.read_hdf(f'./Evaluations/train_6_on_slvenv/SLvEnvEval__slv_207_number_noise_20250109_2DLV_average_less_1_onehalf_day_3.h5', key=f'run_{i}')
    #df = pd.read_hdf(f'./Evaluations/train_6_physicell_evals/train_6_run_3.h5', key=f'run_{i}')

    plot(df, f'LV', scale='linear', truncate=False, ax = ax, c='red')
    ax.set_xlim(0, 600)
    #ax.set_ylim(0, 1.5)
    # divide x ticks by 4
    ticks = ax.get_xticks()
    tick_labels = [int(t/4) for t in ticks]
    ax.set_xticklabels(tick_labels)
    ax.set_yscale('log')
    #fig.savefig('./plots/failing_lv_trajectory_poster.pdf', transparent = True)

#
plt.show()
