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
        fig.savefig(r'fig1_benchmark.pdf', transparent = True)

    plt.show()

def get_mutant_proportions(filename, timesteps=100):
    mutant_proportions = []
    for i in range(timesteps):
        df = pd.read_hdf(filename, key=f'run_{i}')
        initial_size = df['Type 0'][0] + df['Type 1'][0]
        # mutant_proportions.append(df['Type 1'].values[-1]/(df['Type 0'].values[-1] + df['Type 1'].values[-1]))
        nz = df[((df['Type 0'] + df['Type 1']) / initial_size > 1.33)]
        if len(nz) > 0:
            mutant_proportions.append(nz['Type 1'].values[0] / (nz['Type 0'].values[0] + nz['Type 1'].values[0]))
        else:
            mutant_proportions.append(df['Type 1'].values[-1] / (df['Type 0'].values[-1] + df['Type 1'].values[-1]))
    return mutant_proportions

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

    LV_files_list = ['./Evaluations/saved_paper_2d_evals/LvEnvEval_2d_no_treatment.h5',
                     './Evaluations/saved_paper_2d_evals/LvEnvEval_2d_mtd.h5',
                     './Evaluations/saved_paper_2d_evals/LvEnvEval_2d_at100.h5',
                     './Evaluations/saved_paper_2d_evals/LvEnvEval_greatest_agent_run2703_test_x6.h5',
                     ]
    LV_name_list = ['LV No therapy', 'LV MTD', 'LV AT100', 'LV Agent']

    LV_dict = {}
    for i in range(len(LV_files_list)):
        LV_dict[LV_name_list[i]] = get_ttps(LV_files_list[i])

    LV_df = pd.DataFrame(LV_dict)

    # combine the two dataframes
    combined = {}
    mut_prop_dict = {}
    for i in range(len(PC_name_list)):
        combined[LV_name_list[i]] = LV_df[LV_name_list[i]]
        combined[PC_name_list[i]] = PC_df[PC_name_list[i]]
        mut_prop_dict[PC_name_list[i]] = get_mutant_proportions(PC_files_list[i])
    combined_df = pd.DataFrame(combined)

    # box plot the distribution with scatter using seaborn

    sns.histplot(PC_df['PC Agent'], ax=ax, color='red', kde=True, label='PC Agent')
    # horizontal lines at No treat, MTD and at100
    ax.axvline(x=PC_df['PC No therapy'].mean(), color='blue', linestyle='--', label='No therapy')
    ax.axvline(x=PC_df['PC MTD'].mean(), color='green', linestyle='--', label='MTD')
    ax.axvline(x=PC_df['PC AT100'].mean(), color='purple', linestyle='--', label='AT100')
    xa = ax.twinx()
    xa.scatter(PC_df['PC Agent'], mut_prop_dict['PC Agent'], color='red', label='PC Agent')
    xa.set_ylim(0, 0.2)
    xa.set_ylabel('Mutant proportion')

    return

if __name__ == '__main__':
    # setup pwd
    # os.chdir('/Users/saif/Desktop/Serhii/Projects/PhysiLearning')
    os.chdir('/home/saif/Projects/PhysiLearning')
    fig, ax = plt.subplots(figsize=(290 / 72, 160 / 72), constrained_layout=True)
    figure_setup(fig, ax, save_figure = False)