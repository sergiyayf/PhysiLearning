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

def figure_setup(fig, ax, df, save_figure = False, savename = 'test.pdf'):
    plot(fig, ax, df)
    #ax.legend()
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yscale('linear')
    ax[1].set_xlabel('Time to progression', fontsize=6)
    ax[1].set_ylabel('Normalized burden', fontsize=6)
    ax[1].set_xticks([0, 50, 100, 150])
    ax[1].set_xlim(0, 150)
    ax[1].set_ylim(0, 1.55)
    #fig.set_constrained_layout_pads(w_pad=10 / 72, h_pad=10 / 72, hspace=2 / 72, wspace=2 / 72)
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)

    if save_figure:
        fig.savefig(savename, transparent = True)

    plt.show()

def plot(fig, ax, df):

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
    treat = df['Treatment'].values
    # replace 0s that are directly after 1 with 1s
    treat = np.where(treat == 0, np.roll(treat, -1), treat)
    ax[0].fill_between(df.index, 0, 1, where=treat==1, color='orange', label='drug',
    lw=0)
if __name__ == '__main__':
    # setup pwd
    os.chdir('/Users/saif/Desktop/Serhii/Projects/PhysiLearning')
    fig, ax = plt.subplots(2,1, figsize=(130 / 72, 85 / 72), constrained_layout=False,
                           sharex=True, gridspec_kw={'height_ratios': [1, 10]})
    df = pd.read_hdf('./Evaluations/LvEnvEval__n2t4_on_itself1504_n2_t4_l3.h5', key='run_0')
    figure_setup(fig, ax, df, save_figure = True, savename = 'fig2_trajectory_LV_noise_n2t4_lin.pdf')


    fig, ax = plt.subplots(2, 1, figsize=(130 / 72, 85 / 72), constrained_layout=False,
                           sharex=True, gridspec_kw={'height_ratios': [1, 10]})
    df = pd.read_hdf('data/2D_benchmarks/n2_t4_l3/2d_n2_t4_l3_all.h5', key=f'run_7')
    figure_setup(fig, ax, df, save_figure=True, savename='fig2_trajectory_PC_n2t4_lin.pdf')