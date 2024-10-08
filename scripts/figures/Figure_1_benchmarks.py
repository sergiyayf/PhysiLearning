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
    print(ax.get_xticklabels())
    ax.legend()
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)
    # ax.set_yticklabels(ax.get_yticks(), fontsize=6)
    ax.set_ylabel('Time to progression', fontsize=6)
    fig.set_constrained_layout_pads(w_pad=10 / 72, h_pad=10 / 72, hspace=2 / 72, wspace=2 / 72)
    if save_figure:
        fig.savefig(r'one_comp_fig1_benchmark.svg', transparent = True)

    plt.show()


def plot(fig, ax):
    PC_files_list = ['data/2D_benchmarks/no_treatment/2d_no_treatment_all.h5',
                     'data/2D_benchmarks/mtd/2d_mtd_all.h5',
                     'data/2D_benchmarks/at100/2d_at100_all.h5',
                     # 'data/2D_benchmarks/x6/2d_x6_all.h5',
                     'data/2D_benchmarks/one_comp/t12_one_comp/2d_no_noise_run_all.h5',
                     'data/2D_benchmarks/one_comp/with_noise_t12/2d_with_noise_run_all.h5',
                     'data/2D_benchmarks/one_comp/qslv_t12/2d_qslv_run_all.h5',
                     ]
    PC_name_list = ['PC No therapy', 'PC MTD', 'PC AT100', 'PC No noise', 'PC With noise', 'PC Abstracted']

    PC_dict = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])

    PC_df = pd.DataFrame(PC_dict)

    LV_files_list = ['./Evaluations/saved_paper_2d_evals/LvEnvEval_2d_no_treatment.h5',
                     './Evaluations/saved_paper_2d_evals/LvEnvEval_2d_mtd.h5',
                     './Evaluations/saved_paper_2d_evals/LvEnvEval_2d_at100.h5',
                     # './Evaluations/saved_paper_2d_evals/LvEnvEval_greatest_agent_run2703_test_x6.h5',
                     './Evaluations/LvEnvEval_train_envs_no_noise.h5',
                     './Evaluations/LvEnvEval_train_envs_with_noise.h5',
                     './Evaluations/SLvEnvEval_train_envs_qslv.h5',
                     ]
    LV_name_list = ['LV No therapy', 'LV MTD', 'LV AT100', 'LV No noise', 'LV With noise', 'LV Abstracted']

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
    os.chdir('/home/saif/Projects/PhysiLearning')
    fig, ax = plt.subplots(figsize=(290 / 72, 160 / 72), constrained_layout=True)
    figure_setup(fig, ax, save_figure = True)