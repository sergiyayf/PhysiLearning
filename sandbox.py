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
        truncated = df[((df['Type 0'] + df['Type 1'])/initial_size >= 1.5)]
        print(truncated)
        index = truncated.index[0]
        # replace df with zeros after index
        df.loc[index:, 'Type 0'] = 0
        df.loc[index:, 'Type 1'] = 0
        df.loc[index:, 'Treatment'] = 0
    ax.plot(df.index, df['Type 0'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 0')
    ax.plot(df.index, df['Type 1'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1')
    ax.plot(df.index, (df['Type 0'] + df['Type 1'])/(df['Type 0'][0]+df['Type 1'][0]), label='total')
    ax.legend()
    ax.set_title(title)
    ax.set_yscale(scale)
    treat = df['Treatment'].values
    # replace 0s that are directly after 1 with 1s
    #treat = np.where(treat == 0, np.roll(treat, -1), treat)
    ax.fill_between(df.index, 1.0, 1.25, where=treat == 1, color='orange', label='drug',
                       lw=2)
    return ax

def get_ttps(filename, timesteps=100):
    ttps = []
    for i in range(timesteps):
        df = pd.read_hdf(filename, key=f'run_{i}')
        # find the largest index with non-zero Type 0 and Type 1
        initial_size = df['Type 0'][0] + df['Type 1'][0]
        nz = df[((df['Type 0'] + df['Type 1'])/initial_size > 1.5)]
        if len(nz) > 0:
            # append index when type 0 + type 1 is larger than 1.5
            ttps.append(nz.index[0])
        else:
            ttps.append(len(df))
    return ttps

def main():
    PC_files_list = ['data/3D_benchmarks/p62/p62_no_treat/p62_no_treat_all.h5',
                     'data/3D_benchmarks/p62/p62_mtd/p62_mtd_all.h5',
                     'data/3D_benchmarks/p62/p62_at100/p62_at100_all.h5',
                     'data/3D_benchmarks/p62/p62_at100/p62_at100_all.h5',
                     #'data/3D_benchmarks/random/random_all.h5'
                     ]
    PC_name_list = ['PC No therapy', 'PC MTD', 'PC AT100', 'PC AT100_repeat']

    PC_dict = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])

    PC_df = pd.DataFrame(PC_dict)

    LV_files_list = ['./Evaluations/LvEnvEval__3d_lvp62_no_treatment.h5',
                     './Evaluations/LvEnvEval__3d_lvp62_mtd.h5',
                     './Evaluations/LvEnvEval__3d_lvp62_at100.h5',
                     './Evaluations/LvEnvEval__3d_lvp62_fixed_1_4.h5',
                        # './Evaluations/temp/LvEnvEvalfixed_1_4_1_5.h5',
                        # './Evaluations/LvEnvEval_3d_cont_4_rav19022024_run_22_load_4.h5',
                        #'./Evaluations/LvEnvEvalrandom_1_5.h5'
                        ]
    LV_name_list = ['LV No therapy', 'LV MTD', 'LV AT100', 'LV fixed 1.4']

    LV_dict = {}
    for i in range(len(LV_files_list)):
        LV_dict[LV_name_list[i]] = get_ttps(LV_files_list[i])

    LV_df = pd.DataFrame(LV_dict)

    SLV_files_list = ['./Evaluations/SLvEnvEval__3d_slvp62_no_treatment.h5',
                     './Evaluations/SLvEnvEval__3d_slvp62_mtd.h5',
                     './Evaluations/SLvEnvEval__3d_slvp62_at100.h5',
                     './Evaluations/SLvEnvEval__3d_slvp62_fixed_1_4.h5',
                     # './Evaluations/temp/LvEnvEvalfixed_1_4_1_5.h5',
                     # './Evaluations/LvEnvEval_3d_cont_4_rav19022024_run_22_load_4.h5',
                     # './Evaluations/LvEnvEvalrandom_1_5.h5'
                     ]
    SLV_name_list = ['SLV No therapy', 'SLV MTD', 'SLV AT100', 'SLV fixed 1.4']

    SLV_dict = {}
    for i in range(len(SLV_files_list)):
        SLV_dict[SLV_name_list[i]] = get_ttps(SLV_files_list[i])

    SLV_df = pd.DataFrame(SLV_dict)
    # combine the two dataframes
    combined = {}
    for i in range(len(PC_name_list)):
        combined[PC_name_list[i]] = PC_df[PC_name_list[i]]
        combined[LV_name_list[i]] = LV_df[LV_name_list[i]]
        combined[SLV_name_list[i]] = SLV_df[SLV_name_list[i]]
    combined_df = pd.DataFrame(combined)

    # box plot the distribution with scatter using seaborn
    fig, ax = plt.subplots()
    sns.boxplot(data=combined_df, ax=ax, width = 0.3)
    sns.stripplot(data=combined_df, ax=ax, color='black', jitter=0.2, size=2.5)
    # show mean as well
    ax.scatter(combined_df.mean().index, combined_df.mean(), marker='x', color='red', s=50, label='mean')


# df = pd.read_hdf('data/3D_benchmarks/at100/at100_all.h5', key=f'run_3')
# plot(df, 'at100 PC', scale='linear')
#
# df = pd.read_hdf('Evaluations/LvEnvEval_3d_fixed_1_9.h5', key=f'run_0')
# plot(df, 'LV fixed 1.9', scale='linear')
#
# df = pd.read_hdf('data/3D_benchmarks/rl_model_on_PC/rl_model_on_PC_all.h5', key=f'run_0')
# plot(df, 'RL model on PC', scale='linear', truncate=False)
#
# df = pd.read_hdf('data/temp/multp_x6/run_1.h5', key=f'run_0')
# plot(df, 'x6 model on PC x6 ', scale='linear', truncate=False)
#
# df = pd.read_hdf('data/temp/PcEnvEval__s2t5_pc_pat_1_test1504_s2_t5_l3.h5', key=f'run_0')
# plot(df, 's2 agent on 3D p3', scale='linear', truncate=False)
#
# df = pd.read_hdf('data/2D_benchmarks/n2_t4_l3/2d_n2_t4_l3_all.h5', key=f'run_7')
# plot(df, 'PC n2t4', scale='linear', truncate=False)


# df = pd.read_hdf(f'data/3D_benchmarks/p62/p62_at100/p62_at100_all.h5', key=f'run_34')
# plot(df, 'PC at100', scale='linear', truncate=False)
df = pd.read_hdf('data/3D_benchmarks/p62/p62_at100/p62_at100_all.h5', key=f'run_65')
plot(df, 'at100 PC', scale='linear', truncate=False)

df = pd.read_hdf('data/3D_benchmarks/p62/p62_at100/p62_at100_all.h5', key=f'run_65')
plot(df, 'at100 PC', scale='log', truncate=False)

df = pd.read_hdf('Evaluations/LvEnvEval__3d_lvp62_at100.h5', key=f'run_0')
plot(df, 'LV at100', scale='linear', truncate=False)

df = pd.read_hdf('Evaluations/LvEnvEval_3d_lvp62_det_agent1405_check_3d_lv_agent_p_62_t_1.h5', key=f'run_0')
plot(df, 'LV agent lv', scale='linear', truncate=False)

df = pd.read_hdf('Evaluations/SLvEnvEval__3d_slvp62_at100.h5', key=f'run_10')
plot(df, 'SLV at100', scale='linear', truncate=False)

df = pd.read_hdf('Evaluations/SLvEnvEval__3d_slvp62_at100.h5', key=f'run_10')
plot(df, 'SLV at100', scale='log', truncate=False)

main()
plt.show()
