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
    PC_files_list = ['data/2D_benchmarks/multipatient/agent_x6/2d_mult_x6_run_all.h5',
                     'data/2D_benchmarks/multipatient/agent_n2t4/2d_mult_n2t4_run_all.h5',
                     'data/2D_benchmarks/multipatient/agent_s2t5/2d_mult_s2t5_run_all.h5',
                     ]
    PC_name_list = ['PC x6', 'PC n2t4', 'PC s2t5']

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

if __name__ == '__main__':
    # setup pwd
    os.chdir('/Users/saif/Desktop/Serhii/Projects/PhysiLearning')
    fig, ax = plt.subplots(figsize=(175 / 72, 135 / 72), constrained_layout=True)
    figure_setup(fig, ax, save_figure = True)