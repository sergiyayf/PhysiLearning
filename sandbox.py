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
    sus = np.array(df['Type 0'].values )#/ (df['Type 0'][normalize] + df['Type 1'][normalize]))
    res = np.array(df['Type 1'].values )#/ (df['Type 0'][normalize] + df['Type 1'][normalize]))
    # sus = np.array(df['Type 0'].values)*70
    # res = np.array(df['Type 1'].values)*70
    tot = sus + res
    time = df.index / 2
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

    ax.legend()
    ax.set_title(title)
    ax.set_yscale(scale)
    ax.hlines(1.0, 0, time[-1], color='k', linestyle='--')
    treat = np.array(df['Treatment'].values)
    treat = treat[:len(time)]
    # replace 0s that are directly after 1 with 1s
    treat = np.where(treat == 0, np.roll(treat, -1), treat)
    ax.fill_between(time, 0, np.max(tot), where=treat==1, color=treatment_color, label='drug',
    lw=2)

    nc = [7022, 7368, 8799, 10946, 12655, 15305, 16515, 19284, 20086, 24395, 27738]
    pulse = [7245, 7074, 8394, 9172, 8018, 6935, 6292, 6961, 7593, 8577]
    at50 = [7071, 7267, 7954, 8794, 7637, 6239, 4710, 3614, 2413, 2024, 1910, 2417, 3493, 4924, 6307, 7867]
    at100 = [7831, 7793, 9121, 9667, 8485, 6922, 4689, 3607, 2694, 3117, 3797, 5218, 6657, 8613]
    mtd = [6980, 6883, 7888, 8986, 7665, 6079, 4157, 3103, 1823, 1426, 911, 803, 493, 426]
    # ax.scatter(range(len(nc)-1), np.array(nc)[1:], color='black', label='NC')
    # ax.scatter(range(len(pulse)-1), np.array(pulse)[1:], color='blue', label='pulse')
    # ax.scatter(range(len(at50)-1), np.array(at50)[1:], color='red', label='at50')
    # ax.scatter(range(len(at100)-1), np.array(at100)[1:], color='green', label='at100')
    # ax.scatter(range(len(mtd)-1), np.array(mtd)[1:], color='orange', label='mtd')
    #ax.scatter(range(len(b)), np.array(b), color='green', label='B7')
    #ax.scatter(range(len(c)), np.array(c), color='blue', label='C7')
    #ax.scatter(range(len(d)), np.array(d), color='red', label='D7')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell count')
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
fig, ax = plt.subplots()
df = pd.read_hdf('./Evaluations/ArrEnvEval__nc.h5', key='run_0')
plot(df, 'arrest nc', scale='linear', truncate=False, ax = ax, c='black')
fig, ax = plt.subplots()
df = pd.read_hdf('./Evaluations/ArrEnvEval__eat100.h5', key='run_0')
plot(df, 'arrest pulse', scale='linear', truncate=False, ax = ax, c='blue')
fig, ax = plt.subplots()
df = pd.read_hdf('./Evaluations/ArrEnvEval__mtd.h5', key='run_0')
plot(df, 'arrest mtd 8', scale='linear', truncate=False, ax = ax, c='orange')
fig, ax = plt.subplots()
df = pd.read_hdf('./Evaluations/ArrEnvEval__at50.h5', key='run_0')
plot(df, 'arrest at50', scale='linear', truncate=False, ax = ax, c='red')
fig, ax = plt.subplots()
df = pd.read_hdf('./Evaluations/ArrEnvEval__at100.h5', key='run_0')
plot(df, 'arrest at100', scale='linear', truncate=False, ax = ax, c='green')

# df = pd.read_hdf('./Evaluations/LvEnvEval__eat100.h5', key='run_0')
# plot(df, 'LV pulse', scale='linear', truncate=False, ax = ax, c='blue')

# df = pd.read_hdf('./Evaluations/LvEnvEval__on_off.h5', key='run_0')
# plot(df, 'LV pulse', scale='linear', truncate=False, ax = ax, c='c')
#
# df = pd.read_hdf('./Evaluations/ArrEnvEval__on_off.h5', key='run_0')
# plot(df, 'arrest at100', scale='linear', truncate=False, ax = ax, c='cyan')

#ax.set_yscale('log')
main()

plt.show()
