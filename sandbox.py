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

    skip = 6
    normalize = 2
    sus = np.array(df['Type 0'].values[::skip] )#/ (df['Type 0'][normalize] + df['Type 1'][normalize]))
    res = np.array(df['Type 1'].values[::skip] )#/ (df['Type 0'][normalize] + df['Type 1'][normalize]))
    # sus = np.array(df['Type 0'].values)*70
    # res = np.array(df['Type 1'].values)*70
    tot = sus + res
    time = df.index[::skip] / skip
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

    # ax.legend()
    ax.set_title(title)
    ax.set_yscale(scale)
    ax.hlines(1.0, 0, time[-1], color='k', linestyle='--')
    treat = np.array(df['Treatment'].values[::skip])
    treat = treat[:len(time)]
    # replace 0s that are directly after 1 with 1s
    #treat = np.where(treat == 0, np.roll(treat, -1), treat)
    for t in range(len(time)-1):
        if treat[t] == 1:
            ax.axvspan((t-1), t, color=treatment_color)
    #ax.fill_between(time, 0, np.max(tot), where=treat==1, color=treatment_color, label='drug',
    #lw=2)

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
    # ax.legend()
    return ax

def get_ttps(filename, timesteps=10):
    ttps = []
    for i in range(timesteps):
        df = pd.read_hdf(filename, key=f'run_{i}')
        # find the largest index with non-zero Type 0 and Type 1
        initial_size = df['Type 0'][0] + df['Type 1'][0]
        nz = df[((df['Type 1'])/initial_size >= 1.0)]
        tot_idx = df[((df['Type 0'] + df['Type 1'])/initial_size >= 1.5)].index
        if len(nz) > 0:
            # append index when type 0 + type 1 is larger than 1.5
            ttps.append(nz.index[0]/2)
        elif len(tot_idx) > 0:
            ttps.append(tot_idx[0]/2)
        else:
            ttps.append(len(df)/2)
        #ttps.append(df['Type 1'][52]/initial_size)
    return ttps

def main():
    PC_files_list = ['./data/3D_manuals/nc/nc_all.h5',
                    './data/3D_manuals/mtd/mtd_all.h5',
                     './data/3D_manuals/at50/at50_all.h5',
                     './data/3D_manuals/at100/at100_all.h5',
                    # './Evaluations/run_evals/mtd.h5',
                    #  './Evaluations/run_evals/at50.h5',
                    #  './Evaluations/run_evals/at100.h5',
                    #  './Evaluations/run_evals/eat100.h5',
                     #'data/3D_benchmarks/random/random_all.h5'
                     ]
    PC_name_list = ['nc', 'mtd', 'at50', 'at100']

    PC_dict = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])

    PC_df = pd.DataFrame(PC_dict)

    LV_files_list = ['Evaluations/LvEnvEval__nc.h5',
        'Evaluations/LvEnvEval__mtd.h5',
        'Evaluations/LvEnvEval__at50.h5',
        'Evaluations/LvEnvEval__at100.h5',
        # './Evaluations/LvEnvEval__mtd.h5',
        #                 './Evaluations/LvEnvEval__at50.h5',
        #                 './Evaluations/LvEnvEval__at100.h5',
        #                 './Evaluations/LvEnvEval__eat100.h5',
                        #'./Evaluations/LvEnvEvalrandom_1_5.h5'
                        ]
    LV_name_list = ['Lv nc', 'Lv mtd', 'Lv at50', 'Lv at100']

    LV_dict = {}
    for i in range(len(LV_files_list)):
        LV_dict[LV_name_list[i]] = get_ttps(LV_files_list[i])

    LV_df = pd.DataFrame(LV_dict)

    # combine the two dataframes
    combined = {}
    for i in range(len(PC_name_list)):
        combined[PC_name_list[i]] = PC_df[PC_name_list[i]]
        #combined[LV_name_list[i]] = LV_df[LV_name_list[i]]
    combined_df = pd.DataFrame(combined)

    # box plot the distribution with scatter using seaborn
    fig, ax = plt.subplots()
    sns.boxplot(data=combined_df, ax=ax, width = 0.3)
    sns.stripplot(data=combined_df, ax=ax, color='black', jitter=0.2, size=2.5)
    # show mean as well
    ax.scatter(combined_df.mean().index, combined_df.mean(), marker='x', color='red', s=50, label='mean')


# fig, ax = plt.subplots()
# df = pd.read_hdf('./Evaluations/3d_at50.h5', key='run_0')
# plot(df, '3d at50', scale='linear', truncate=False, ax = ax, c='red')
# df = pd.read_hdf('./Evaluations/LvEnvEval__3D_at50.h5', key='run_0')
# plot(df, 'Lv 3D at50', scale='linear', truncate=False, ax = ax, c='red')
#
# fig, ax = plt.subplots()
# df = pd.read_hdf('./Evaluations/LvEnvEval__3D_deep.h5', key='run_0')
# plot(df, 'Lv 3D deep', scale='linear', truncate=False, ax = ax, c='red')
# fig, ax = plt.subplots()
# df = pd.read_hdf('./Evaluations/LvEnvEval_3D_deep20241031_3D_hypers_3.h5', key='run_0')
# plot(df, 'Lv 3D agnt', scale='linear', truncate=False, ax = ax, c='red')
#
# for i in range(1):
#     fig, ax = plt.subplots()
#     agnt = './data/20241202_day_run_9_t8_and_t10_agents/agent_r9t10/Evaluations/PcEnvEval_agnt_r9t1020241202_2DLV_10.h5'
#     #agnt = './data/20241202_day_run_9_t8_and_t10_agents/agent_r9t8/Evaluations/PcEnvEval_agnt_r9t820241202_2DLV_8.h5'
#     df = pd.read_hdf(agnt, key=f'run_{i}')
#     plot(df, f'agnt10 {i}', scale='linear', truncate=False, ax = ax, c='red')

#
# fig, ax = plt.subplots()
# df = pd.read_hdf('./Evaluations/LvEnvEval__e_140-060.h5', key=f'run_1')
# plot(df, f'LV', scale='linear', truncate=False, ax = ax, c='red')
#
# fig, ax = plt.subplots()
# df = pd.read_hdf('./Evaluations/SLvEnvEval__e_140-060.h5', key=f'run_1')
# plot(df, f'SLV', scale='linear', truncate=False, ax = ax, c='red')
#
# fig, ax = plt.subplots()
# df = pd.read_hdf('./Evaluations/LvEnvEval__e_110-090.h5', key=f'run_1')
# plot(df, f'LV', scale='linear', truncate=False, ax = ax, c='red')
#
fig, axs = plt.subplots(10,10)
for i in range(1,11):
    for j in range(0,10):
        # fig, ax = plt.subplots()

        #df = pd.read_hdf(f'./Evaluations/lv_weekend_1001_trains/train_5/LvEnvEval__agnt_20250110_2DLV_improve_try_{i}.h5', key=f'run_{j}')
        # df = pd.read_hdf(f'./Evaluations/0901_onehalf_day_6/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_{i}.h5', key=f'run_{j}')
        df = pd.read_hdf(f'./Evaluations/train_6_physicell_evals/train_6_run_{i}.h5', key=f'run_{j}')
        #df = pd.read_hdf(f'./Evaluations/train_6_on_slvenv/SLvEnvEval__slv_20250109_2DLV_average_less_1_onehalf_day_{i}.h5', key=f'run_{j}')
        # df = pd.read_hdf(f'./Evaluations/1402_lvs_evals/LvEnvEval__20250206_lv_1_{i}.h5',
        #df=pd.read_hdf(f'./Evaluations/1402_pcs_evals/run_{i}.h5',
        # df = pd.read_hdf(f'./Evaluations/2002_pc_evals_of_slvs/run_{i}.h5',
        #                 key=f'run_{j}')
        ax = axs[i-1, j]
        plot(df, f'PC Daily', scale='linear', truncate=False, ax = ax, c='red')
        # calculate average totatl cell count
        tot = df['Type 0'] + df['Type 1']
        tot = tot[tot > 0]
        print(np.mean(tot))
        max_sens = np.max(df['Type 0'])
        min_sens = np.min(df['Type 0'])
        min_tot = np.min(tot)
        ax.set_title(f'C: {np.mean(tot):.2f}, A: {min_tot:.2f}')
        ax.set_xlim(0, 120)
        #fig.savefig(f'./plots/lucky_4_pc_{i}.pdf')
        #ax.set_yscale('log')
        fig.suptitle(f'pc')
#
# for i in range(6):
#     fig, ax = plt.subplots()
#     df = pd.read_hdf('./Evaluations/pcs_09_6/PcEnvEval_run20250109_2DLV_average_less_1_onehalf_day_4.h5', key=f'run_{i}')
#     plot(df, f'PC', scale='linear', truncate=False, ax = ax, c='red')
#     ax.set_xlim(0, 100)
#ax.set_yscale('log')
# ax[0].set_xlim(0, 300)
# for i in range(1,6):
#     df = pd.read_hdf(f'./Evaluations/ptb{i}.h5', key=f'run_0')
#     plot(df, f'perturbed {i}', scale='linear', truncate=False, ax = ax[i], c='red')
#     ax[i].set_xlim(0, 300)
# fig, ax = plt.subplots()
# df = pd.read_hdf('./Evaluations/SLvEnvEval_r9t820241202_2DLV_8.h5', key=f'run_1')
# plot(df, f'original', scale='linear', truncate=False, ax = ax, c='red')
# ax.set_xlim(0, 300)
#main()
plt.show()
