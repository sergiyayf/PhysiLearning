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

def get_ttps(filename, timesteps=20):
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
        fig.savefig(r'CT_vs_AT_si.pdf', transparent = True)

    plt.show()


def plot(fig, ax):
    at50 = f'./data/29112024_2d_manuals/at50/PcEnvEval_job_1388687120241129_2D_manuals_at50.h5'
    mtd = f'./data/29112024_2d_manuals/mtd/PcEnvEval_job_1388687220241129_2D_manuals_mtd.h5'

    ttps_at50 = get_ttps(at50)
    ttps_mtd = get_ttps(mtd)
    print(ttps_at50)
    print(ttps_mtd)

    # compare averages
    benefit = (np.mean(ttps_at50) - np.mean(ttps_mtd))/np.mean(ttps_mtd)
    print(f'Benefit of AT50 over MTD: {benefit}')
    return

if __name__ == '__main__':
    # setup pwd
    os.chdir('/home/saif/Projects/PhysiLearning')

    fig, ax = plt.subplots(figsize=(290 / 72, 160 / 72), constrained_layout=True)
    figure_setup(fig, ax, save_figure = False)