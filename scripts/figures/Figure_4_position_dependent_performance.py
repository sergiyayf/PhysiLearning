import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
ex = {'font.size': 6,
          'font.weight': 'normal',
          'font.family': 'sans-serif'}
plt.rcParams.update(ex)
mpl.rcParams['pdf.fonttype'] = 42  # to make text editable in pdf output
mpl.rcParams['font.sans-serif'] = ['Arial']  # to make it Arial

def get_ttps(filename, timesteps=100):
    ttps = []
    for i in range(timesteps):
        df = pd.read_hdf(filename, key=f'run_{i}')
        # find the largest index with non-zero Type 0 and Type 1
        initial_size = df['Type 0'][0] + df['Type 1'][0]
        nz = df[((df['Type 0'] + df['Type 1'])/initial_size > 1.33)]
        if len(nz) > 0:
            # append index when type 0 + type 1 is larger than 1.5
            ttps.append(nz.index[0])
        else:
            ttps.append(len(df))

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
    ax.set_ylabel('Time to progression', fontsize=6)
    ax.set_xlabel('Treatment', fontsize=6)

    # ax.hlines(38, 0, max(ax.get_xticks()), color='k', linestyle='--', lw=1.5, label='MTD')
    # ax.hlines(72.5, 0, max(ax.get_xticks()), color='r', linestyle='--', lw=1.5, label='AT100')
    ax.legend()

    fig.set_constrained_layout_pads(w_pad=10 / 72, h_pad=10 / 72, hspace=2 / 72, wspace=2 / 72)
    if save_figure:
        fig.savefig(r'fig4_boxplot.pdf', transparent = True)

    plt.show()


def plot(fig, ax):
    PC_files_list = ['data/2D_benchmarks/multipatient/parabolic_agent_different_positions/p1/2d_run_all.h5',
                     'data/2D_benchmarks/multipatient/parabolic_agent_different_positions/p26/2d_run_all.h5',
                     'data/2D_benchmarks/multipatient/parabolic_agent_different_positions/p15/2d_run_all.h5',
                     'data/2D_benchmarks/parabolic_t1/2d_parabolic_run_all.h5',
                     'data/2D_benchmarks/multipatient/parabolic_agent_different_positions/p78/2d_run_all.h5',
                     'data/2D_benchmarks/multipatient/parabolic_agent_different_positions/p71/2d_run_all.h5',
                     'data/2D_benchmarks/multipatient/parabolic_agent_different_positions/p57/2d_run_all.h5',
                     'data/2D_benchmarks/multipatient/parabolic_agent_different_positions/p65/2d_run_all.h5',
                     ]
    PC_name_list = ['0', '27', '52', '71', '98', '136', '198', 'none']

    PC_dict = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])

    PC_df = pd.DataFrame(PC_dict)

    # box plot the distribution with scatter using seaborn

    b = sns.boxplot(data=PC_df, ax=ax, width=0.3, fliersize=1.5, linewidth=1)
    sns.stripplot(data=PC_df, ax=ax, color='black', jitter=0.2, size=1.5, alpha=0.5)
    # show mean as well
    ax.scatter(PC_df.mean().index, PC_df.mean(), marker='x', color='red', s=20, label='mean')

    return b

def plot_trajectory(fig, ax, df):

    #ax.plot(df.index, df['Type 0'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 0', color='b')
    #ax.plot(df.index, df['Type 1'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1', color='r')
    total = df['Type 0'].values + df['Type 1'].values
    # where = 0 set to nan
    total = np.where(total == 0, np.nan, total)
    type0 = np.where(df['Type 0'].values == 0, np.nan, df['Type 0'].values)
    type1 = np.where(df['Type 1'].values == 0, np.nan, df['Type 1'].values)
    ax[1].plot(df.index, type0/total[0], label='Type 0', color='b')
    ax[1].plot(df.index, type1/total[0], label='Type 1', color='r')
    ax[1].plot(df.index, total/total[0], label='total', color='k')
    ax[1].axvline(x=7, color='k', linestyle='--',label='No treatment')
    ax[1].axvline(x=37, color='orange', linestyle='--',label='MTD')
    ax[1].axvline(x=73, color='green', linestyle='--',label='AT100')

    treat = df['Treatment'].values
    # replace 0s that are directly after 1 with 1s
    treat = np.where(treat == 0, np.roll(treat, -1), treat)
    ax[0].fill_between(df.index, 0, 1, where=treat==1, color='orange', label='drug',
    lw=0)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yscale('linear')
    ax[1].set_xlabel('Time to progression', fontsize=6)
    ax[1].set_ylabel('Normalized burden', fontsize=6)
    ax[1].set_xticks([0, 50, 100, 150])
    ax[1].set_xlim(0, 150)
    ax[1].set_ylim(0, 1.55)
    # fig.set_constrained_layout_pads(w_pad=10 / 72, h_pad=10 / 72, hspace=2 / 72, wspace=2 / 72)
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)

if __name__ == '__main__':
    # setup pwd
    os.chdir('/home/saif/Projects/PhysiLearning')
    fig, ax = plt.subplots(figsize=(175 / 72, 135 / 72), constrained_layout=True)
    figure_setup(fig, ax, save_figure = False)

    # plot trajectories of multiple runs
    for i in range(1):
        df = pd.read_hdf('data/2D_benchmarks/parabolic_t1/2d_parabolic_run_all.h5', key=f'run_{i}')
        fig, ax = plt.subplots(2,1, constrained_layout=False, figsize=(200 / 72, 80 / 72),sharex=True, gridspec_kw={'height_ratios': [1, 10]})
        plot_trajectory(fig, ax, df)

    plt.show()