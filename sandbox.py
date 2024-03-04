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
        truncated = df[((df['Type 0'] + df['Type 1'])/initial_size >= 2.0)]
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
    treat = np.where(treat == 0, np.roll(treat, 1), treat)
    ax.fill_between(df.index, 1, 1.250, where=treat==1, color='orange', label='drug',
    lw=2)
    return ax

def get_ttps(filename, timesteps=50):
    ttps = []
    for i in range(timesteps):
        df = pd.read_hdf(filename, key=f'run_{i}')
        # find the largest index with non-zero Type 0 and Type 1
        initial_size = df['Type 0'][0] + df['Type 1'][0]
        nz = df[((df['Type 0'] + df['Type 1'])/initial_size > 2.4)]
        if len(nz) > 0:
            # append index when type 0 + type 1 is larger than 1.5
            ttps.append(nz.index[0])
        else:
            ttps.append(0)
    return ttps

def main():
    PC_files_list = ['data/3D_benchmarks/no_treatment/no_treatment_all.h5',
                     'data/3D_benchmarks/mtd/mtd_all.h5',
                     'data/3D_benchmarks/at100/at100_all.h5',
                     'data/3D_benchmarks/fixed_2_25/fixed_2_25_all.h5',
                     'data/3D_benchmarks/rl_model_on_PC/rl_model_on_PC_all.h5',
                     #'data/3D_benchmarks/random/random_all.h5'
                     ]
    PC_name_list = ['PC No therapy', 'PC MTD', 'PC AT100', 'PC fixed 1.4', 'PC RL model']

    PC_dict = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])

    PC_df = pd.DataFrame(PC_dict)

    LV_files_list = ['./Evaluations/temp/LvEnvEvalno_treatment_1_5.h5',
                        './Evaluations/temp/LvEnvEvalmtd_1_5.h5',
                        './Evaluations/temp/LvEnvEvalat100_1_5.h5',
                        './Evaluations/temp/LvEnvEvalfixed_1_4_1_5.h5',
                        './Evaluations/LvEnvEval_3d_cont_4_rav19022024_run_22_load_4.h5',
                        #'./Evaluations/LvEnvEvalrandom_1_5.h5'
                        ]
    LV_name_list = ['LV No therapy', 'LV MTD', 'LV AT100', 'LV Fixed 1.4', 'RL']

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


df = pd.read_hdf('data/3D_benchmarks/at100/at100_all.h5', key=f'run_3')
plot(df, 'at100 PC', scale='linear')

df = pd.read_hdf('Evaluations/LvEnvEval_3d_fixed_1_9.h5', key=f'run_0')
plot(df, 'LV fixed 1.9', scale='linear')
#
# df = pd.read_hdf('data/3D_benchmarks/new_mtd/mtd_all.h5', key=f'run_0')
# plot(df, 'mtd PC')
# 
# df = pd.read_hdf('Evaluations/LvEnvEvalat100_1_5.h5', key=f'run_0')
# plot(df, 'LV at100', scale='log')

# df = pd.read_hdf('Evaluations/LvEnvEval_raven_cont16022024_3D_LV_raven_load.h5', key=f'run_0')
# plot(df, 'LV raven cont')
# #
# df = pd.read_hdf('Evaluations/LvEnvEval_raven_run_815022024_3D_LV_raven_load.h5', key=f'run_10')
# plot(df, 'LV policy nt')

# df = pd.read_hdf('data/raven_run_logs/new_run_22/Evaluations/LvEnvEval_raven_22_run16022024_3D_LV_raven_rew_0.h5','run_20')
# plot(df, 'LV policy 22')
#
# df = pd.read_hdf('data/raven_run_logs/new_run_23/Evaluations/LvEnvEval_raven_23_run16022024_3D_LV_raven_rew_0.h5','run_20')
# plot(df, 'LV policy 23')
#
# df = pd.read_hdf('data/raven_run_logs/cont_22_run_4/Evaluations/LvEnvEval_cont_4_rav19022024_run_22_load_4.h5','run_0')
# plot(df, 'LV cont 22 1 ')
#
# df = pd.read_hdf('data/raven_run_logs/cont_22_run_4/Evaluations/LvEnvEval_cont_4_rav19022024_run_22_load_4.h5','run_20')
# plot(df, 'LV cont 22 2')


main()
plt.show()
