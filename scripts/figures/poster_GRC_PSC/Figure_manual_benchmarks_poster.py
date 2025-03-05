import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
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

def figure_setup(fig, ax, save_figure = False):
    # set plot font and fontsizes
    ex = {'font.size': 14,
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
        fig.savefig(r'fig1_benchmark.pdf', transparent = True)

    plt.show()

def jitter(x0, y, ax=None, s=0.1, c='b', alpha=0.5):
    kde = gaussian_kde(y)
    density = kde(y)

    jitter = np.random.normal(0, s, len(y))
    x_vals = x0 + (density * jitter * 1)
    ax.scatter(x_vals, y, color=c, alpha=alpha, s=10)

def plot(fig, ax):
    PC_files_list = ['./Evaluations/PcEnvEval_job_1504085420250116_2D_manuals_nc.h5',
                     './Evaluations/PcEnvEval_job_1514108420251016_2D_manuals_mtd.h5',
                     './Evaluations/PcEnvEval_job_1514097320250116_2D_manuals_at50.h5',
                     './Evaluations/PcEnvEval_job_1514097320250116_2D_manuals_at50.h5',
                     ]
    PC_name_list = ['PC No therapy', 'PC MTD', 'PC AT50', 'PC Agent']

    PC_dict = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])

    PC_df = pd.DataFrame(PC_dict)

    LV_files_list = ['./Evaluations/LvEnvEval__nc.h5',
                     './Evaluations/LvEnvEval__mtd.h5',
                     './Evaluations/LvEnvEval__at50.h5',
                     # './Evaluations/saved_paper_2d_evals/LvEnvEval_2d_no_treatment.h5',
                     # './Evaluations/saved_paper_2d_evals/LvEnvEval_2d_mtd.h5',
                     # './Evaluations/saved_paper_2d_evals/LvEnvEval_2d_at100.h5',
                     'Evaluations/train_6_lv_evals/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_1.h5',
                     ]
    LV_name_list = ['LV No therapy', 'LV MTD', 'LV AT50', 'LV Agent']

    LV_dict = {}
    for i in range(len(LV_files_list)):
        LV_dict[LV_name_list[i]] = get_ttps(LV_files_list[i])

    LV_df = pd.DataFrame(LV_dict)

    # combine the two dataframes
    combined = {}
    for i in range(len(PC_name_list)):
        combined[LV_name_list[i]] = LV_df[LV_name_list[i]]
        combined[PC_name_list[i]] = PC_df[PC_name_list[i]]
    combined_df = pd.DataFrame(combined)

    # box plot the distribution with scatter using seaborn

    b = sns.boxplot(data=combined_df, ax=ax, width=0.3, fliersize=1.5, linewidth=1)
    sns.stripplot(data=combined_df, ax=ax, color='black', jitter=0.2, size=1.5, alpha=0.5)
    # show mean as well
    ax.scatter(combined_df.mean().index, combined_df.mean(), marker='x', color='red', s=20, label='mean')

    return

if __name__ == '__main__':
    # setup pwd
    # os.chdir('/Users/saif/Desktop/Serhii/Projects/PhysiLearning')
    os.chdir('/')
    fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
    figure_setup(fig, ax, save_figure = False)