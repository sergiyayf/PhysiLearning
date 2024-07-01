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
    PC_files_list = ['data/2D_benchmarks/no_treatment/2d_no_treatment_all.h5',
                     'data/2D_benchmarks/mtd/2d_mtd_all.h5',
                     'data/2D_benchmarks/at100/2d_at100_all.h5',
                     'data/2D_benchmarks/x6/2d_x6_all.h5',
                     ]
    PC_name_list = ['PC No therapy', 'PC MTD', 'PC AT100', 'PC Agent']

    PC_dict = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])

    PC_df = pd.DataFrame(PC_dict)

    LV_files_list = ['./Evaluations/SLvEnvEval__nt_e.h5',
                     './Evaluations/SLvEnvEval__mtd_e.h5',
                     './Evaluations/SLvEnvEval__at100_e.h5',
                     # './Evaluations/saved_paper_2d_evals/LvEnvEval_2d_no_treatment.h5',
                     # './Evaluations/saved_paper_2d_evals/LvEnvEval_2d_mtd.h5',
                     # './Evaluations/saved_paper_2d_evals/LvEnvEval_2d_at100.h5',
                     './Evaluations/saved_paper_2d_evals/LvEnvEval_greatest_agent_run2703_test_x6.h5',
                     ]
    LV_name_list = ['LV No therapy', 'LV MTD', 'LV AT100', 'LV Agent']

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

    x_0 = LV_df['LV No therapy'].values[0]
    y = PC_df['PC No therapy'].values
    jitter(x_0, y, ax=ax, c='b', s=0.5)
    x_1 = LV_df['LV MTD'].values[0]
    y = PC_df['PC MTD'].values
    jitter(x_1, y, ax=ax, c='r', s=10)
    x_2 = LV_df['LV AT100'].values[0]
    y = PC_df['PC AT100'].values
    jitter(x_2, y, ax=ax, c='g', s=30)
    ax.boxplot([PC_df['PC No therapy'].values, PC_df['PC MTD'].values, PC_df['PC AT100'].values],
               positions=[x_0, x_1, x_2], widths=10)
    # plot diagonal
    ax.plot([0, 250], [0, 250], 'k--')
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 170)
    ax.set_xlabel('Time to progression LV')
    ax.set_ylabel('Time to progression PC')
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xticklabels([0, 20, 40, 60, 80])

    # b = sns.boxplot(data=combined_df, ax=ax, width=0.3, fliersize=1.5, linewidth=1)
    # sns.stripplot(data=combined_df, ax=ax, color='black', jitter=0.2, size=1.5, alpha=0.5)
    # # show mean as well
    # ax.scatter(combined_df.mean().index, combined_df.mean(), marker='x', color='red', s=20, label='mean')

    return

if __name__ == '__main__':
    # setup pwd
    # os.chdir('/Users/saif/Desktop/Serhii/Projects/PhysiLearning')
    os.chdir('/home/saif/Projects/PhysiLearning')
    fig, ax = plt.subplots(figsize=(290 / 72, 160 / 72), constrained_layout=True)
    figure_setup(fig, ax, save_figure = False)