import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def plot(df, title, scale='linear', truncate=False):
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

    skip = 2
    normalize = 2
    sus = np.array(df['Type 0'].values )#/ (df['Type 0'][normalize] + df['Type 1'][normalize]))
    res = np.array(df['Type 1'].values )#/ (df['Type 0'][normalize] + df['Type 1'][normalize]))
    # sus = np.array(df['Type 0'].values)*70
    # res = np.array(df['Type 1'].values)*70
    tot = sus + res
    time = df.index / 2

    ax.plot(time[::skip], sus[::skip], color='g')
    ax.plot(time[::skip], res[::skip], color='r')
    ax.plot(time[::skip], tot[::skip], color='black', label='LV')

    ax.legend()
    ax.set_title(title)
    ax.set_yscale(scale)
    ax.hlines(1.0, 0, time[-1], color='k', linestyle='--')
    treat = df['Treatment'].values
    # replace 0s that are directly after 1 with 1s
    treat = np.where(treat == 0, np.roll(treat, -1), treat)
    ax.fill_between(time, 0, 200, where=treat==1, color='orange', label='drug',
    lw=2)

    b = np.array([6322, 7215, 8159, 8246, 6563, 5068, 5393, 6203, 6695, 8155, 10244, 12520, 13013])
    d = np.array([6589, 7531, 8595, 8662, 7559, 6745, 6539, 7280, 7845, 8666, 9837, 11615, 11724])
    c = np.array([5845, 6865, 7957, 8014, 6781, 5245, 5274, 6242, 6550, 7790, 9647, 11916, 12452])
    #ax.scatter(range(len(b)), np.array(b), color='green', label='B7')
    #ax.scatter(range(len(c)), np.array(c), color='blue', label='C7')
    #ax.scatter(range(len(d)), np.array(d), color='red', label='D7')
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized cell count')
    ax.legend()
    return ax

def get_ttps(filename, timesteps=10):
    ttps = []
    for i in range(timesteps):
        df = pd.read_hdf(filename, key=f'run_{i}')
        # find the largest index with non-zero Type 0 and Type 1
        initial_size = df['Type 0'][2] + df['Type 1'][2]
        nz = df[((df['Type 1'])/initial_size > 1.0)]
        # if len(nz) > 0:
        #     # append index when type 0 + type 1 is larger than 1.5
        #     ttps.append(nz.index[0]/2)
        #
        # else:
        #     ttps.append(len(df)/2)
        ttps.append(df['Type 1'][52]/initial_size)
    return ttps

def main():
    PC_files_list = ['./data/agnt_1_3/Evaluations/PcEnvEval_a1_320240927_1_3.h5',
                    './data/agnt_1_6/Evaluations/PcEnvEval_a1_620240927_1_6.h5',
                    './data/agnt_1_10/Evaluations/PcEnvEval_a1_1020240927_1_10.h5',
                    './data/agnt_1_11/Evaluations/PcEnvEval_a1_1120240927_1_11.h5',
                    './data/agnt_1_14/Evaluations/PcEnvEval_a1_1420240927_1_14.h5',
                    # './Evaluations/run_evals/mtd.h5',
                    #  './Evaluations/run_evals/at50.h5',
                    #  './Evaluations/run_evals/at100.h5',
                    #  './Evaluations/run_evals/eat100.h5',
                     #'data/3D_benchmarks/random/random_all.h5'
                     ]
    PC_name_list = ['PC agnt 1_3', 'PC agnt 1_6', 'PC agnt 1_10', 'PC agnt 1_11', 'PC agnt 1_14']

    PC_dict = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])

    PC_df = pd.DataFrame(PC_dict)

    LV_files_list = ['Evaluations/scale_t_1_lv_agents/LvEnvEval__agnt_20240927_1_3.h5',
        'Evaluations/scale_t_1_lv_agents/LvEnvEval__agnt_20240927_1_6.h5',
        'Evaluations/scale_t_1_lv_agents/LvEnvEval__agnt_20240927_1_10.h5',
        'Evaluations/scale_t_1_lv_agents/LvEnvEval__agnt_20240927_1_11.h5',
        'Evaluations/scale_t_1_lv_agents/LvEnvEval__agnt_20240927_1_14.h5',
        # './Evaluations/LvEnvEval__mtd.h5',
        #                 './Evaluations/LvEnvEval__at50.h5',
        #                 './Evaluations/LvEnvEval__at100.h5',
        #                 './Evaluations/LvEnvEval__eat100.h5',
                        #'./Evaluations/LvEnvEvalrandom_1_5.h5'
                        ]
    LV_name_list = ['LV agnt 1_3', 'LV agnt 1_6', 'LV agnt 1_10', 'LV agnt 1_11', 'LV agnt 1_14']

    LV_dict = {}
    for i in range(len(LV_files_list)):
        LV_dict[LV_name_list[i]] = get_ttps(LV_files_list[i])

    LV_df = pd.DataFrame(LV_dict)

    # combine the two dataframes
    combined = {}
    for i in range(len(PC_name_list)):
        combined[PC_name_list[i]] = PC_df[PC_name_list[i]]
        combined[LV_name_list[i]] = LV_df[LV_name_list[i]]
    combined_df = pd.DataFrame(combined)

    # box plot the distribution with scatter using seaborn
    fig, ax = plt.subplots()
    sns.boxplot(data=combined_df, ax=ax, width = 0.3)
    sns.stripplot(data=combined_df, ax=ax, color='black', jitter=0.2, size=2.5)
    # show mean as well
    ax.scatter(combined_df.mean().index, combined_df.mean(), marker='x', color='red', s=50, label='mean')


# df_metld = pd.read_hdf('./Evaluations/MeltdEnvEval__test_meltd_at100.h5', key='run_0')
# plot(df_metld, 'Meltd at100', scale='linear', truncate=False)
#
# df_m = pd.read_hdf('./Evaluations/meltd/MeltdEnvEval_1006_2d_meltd_l2.h5', key='run_0')
# plot(df_m, 'Meltd 1006 2d l2', scale='linear', truncate=False)
#
# df_metld = pd.read_hdf('./Evaluations/MeltdEnvEval__test_meltd_eat100.h5', key='run_0')
# plot(df_metld, 'Meltd eat100', scale='linear', truncate=False)
#
# df_pc_mtd = pd.read_hdf('./Evaluations/run_evals/mtd.h5', key='run_0')
# plot(df_pc_mtd, 'PC MTD', scale='linear', truncate=False)

# df = pd.read_hdf('./Evaluations/scale_t_1_lv_agents/LvEnvEval__agnt_20240927_1_10.h5', key='run_0')
# plot(df, 'LV agnt 1_10', scale='linear', truncate=False)
#
# df = pd.read_hdf('./Evaluations/scale_t_1_lv_agents/LvEnvEval__agnt_20240927_1_6.h5', key='run_0')
# plot(df, 'LV agnt 1_6', scale='linear', truncate=False)

df = pd.read_hdf('./Evaluations/LvEnvEval__mtd.h5', key='run_0')
plot(df, 'LV MTD', scale='linear', truncate=False)

main()

plt.show()
