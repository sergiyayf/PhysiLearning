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
        truncated = df[((df['Type 0'] + df['Type 1'])/initial_size >= 1.3)]
        print(truncated)
        if len(truncated) > 0:
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
    treat = np.where(treat == 0, np.roll(treat, 1), treat)
    ax.fill_between(df.index, 1, 1.250, where=treat==1, color='orange', label='drug',
    lw=2)
    return ax

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

def main():
    PC_files_list = ['data/2D_benchmarks/no_treatment/2d_no_treatment_all.h5',
                     'data/2D_benchmarks/mtd/2d_mtd_all.h5',
                     'data/2D_benchmarks/at50/2d_at50_all.h5',
                     'data/2D_benchmarks/at100/2d_at100_all.h5',
                     # 'data/2D_benchmarks/fixed_1_1/2d_fixed_1_1_all.h5',
                     # 'data/2D_benchmarks/fixed_1_2/2d_fixed_1_2_all.h5',
                     #'data/2D_benchmarks/fixed_1_25/2d_fixed_1_25_all.h5',
                     'data/2D_benchmarks/x6/2d_x6_all.h5',
                     # 'data/2D_benchmarks/x7_t6/2d_x7_t6_all.h5',
                     # 'data/2D_benchmarks/x8_t3/2d_x8_t3_all.h5',
                     #
                     'data/2D_benchmarks/slvenv_agent_1/2d_slvenv_agent_1_all.h5',
                     'data/2D_benchmarks/x8_t2/2d_x8_t2_all.h5',
                     #'./Evaluations/LvEnvEval_ref72803_x7_noise_refine_7_punish20.h5',
                     #'data/2D_benchmarks/random/2d_random_all.h5'
                     ]
    PC_name_list = ['PC No therapy', 'PC MTD', 'PC AT50', 'PC AT100', #'PC fixed 1.1', 'PC fixed 1.2',
                    'PC x6', 'PC slvenv 1', 'PC x8 t2']

    PC_dict = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])

    PC_df = pd.DataFrame(PC_dict)

    LV_files_list = ['./Evaluations/LvEnvEval_2d_no_treatment.h5',
                        './Evaluations/LvEnvEval_2d_mtd.h5',
                        './Evaluations/LvEnvEval_2d_at50.h5',
                        './Evaluations/LvEnvEval_2d_at100.h5',
                        # './Evaluations/LvEnvEval_2d_fixed_1_1.h5',
                        # './Evaluations/LvEnvEval_2d_fixed_1_2.h5',
                        # './Evaluations/LvEnvEval_2d_fixed_1_25.h5',
                        './Evaluations/LvEnvEval_greatest_agent_run2703_test_x6.h5',
                        # './Evaluations/LvEnvEval_x8_t2_l5_on_lvnoise2803_x8_cobra_t2_5.h5',
                        #
                        './Evaluations/LvEnvEval_x8_t7_l3_on_lvnoise3103_x8_cobra_t7_load_3.h5',
                        './Evaluations/LvEnvEval_x8_t2_l5_on_lvnoise2803_x8_cobra_t2_5.h5',
                        #'./Evaluations/LvEnvEval_2d_random.h5'

                        ]
    LV_name_list = ['LV No therapy', 'LV MTD', 'LV AT50', 'LV AT100',# 'LV fixed 1.1', 'LV fixed 1.2',
                    'LV x6 RL', 'LV x8 t7', 'LV x8 t2']

    LV_dict = {}
    for i in range(len(LV_files_list)):
        LV_dict[LV_name_list[i]] = get_ttps(LV_files_list[i])

    LV_df = pd.DataFrame(LV_dict)

    SLV_files_list = ['./Evaluations/SLvEnvEval__no_treatment.h5',
                        './Evaluations/SLvEnvEval__mtd.h5',
                        './Evaluations/SLvEnvEval__at50.h5',
                        './Evaluations/SLvEnvEval__at100.h5',
                        './Evaluations/SLvEnvEval__fixed_1_1.h5',
                        './Evaluations/SLvEnvEval__fixed_1_2.h5',
                        './Evaluations/SLvEnvEval__slvenv_agent_on_itself_20304_slvenv_train_try.h5',
                        # './Evaluations/SLvEnvEval__2d_slvenv_fixed_1_25.h5',
                        # './Evaluations/SLvEnvEval__2d_slvenv_rand.h5'
                        ]
    SLV_name_list = ['SLV No therapy', 'SLV MTD', 'SLV AT50', 'SLV AT100', 'SLV fixed 1.1',
                     'SLV fixed 1.2', 'SLV agent on itself']

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

    return combined_df


# df = pd.read_hdf('./Evaluations/LvEnvEval_2d_fixed_1_2.h5', key=f'run_0')
# plot(df, 'LV fixed 1.2', scale='linear', truncate=False)
# #
# df = pd.read_hdf('./data/2D_benchmarks/fixed_1_2/2d_fixed_1_2_all.h5', key=f'run_0')
# plot(df, 'PC fixed 1.2', scale='linear', truncate=True)
# #
#
# df = pd.read_hdf('./Evaluations/LvEnvEval_job_30162d_fixed_1_2_noised.h5', key=f'run_0')
# plot(df, 'LV fixed 1.2 with noise', scale='linear', truncate=False)
sims = range(1, 7)
# for sim in sims:
#     df = pd.read_hdf('data/2D_benchmarks/x8_t3/2d_x8_t3_all.h5', key=f'run_{sim}')
#     plot(df, f'x8 PC {sim}', scale='log', truncate=False)
#
# for sim in sims:
#     df = pd.read_hdf('./data/2D_benchmarks/x6/2d_x6_all.h5', key=f'run_{sim}')
#     plot(df, f'x6 {sim}', scale='linear', truncate=False)

for sim in sims:
    df = pd.read_hdf('./data/2D_benchmarks/slvenv_agent_1/2d_slvenv_agent_1_all.h5', key=f'run_{sim}')
    plot(df, f'slvenv PC {sim}', scale='log', truncate=False)


combined_df = main()
plt.show()
