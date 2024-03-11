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

def get_ttps(filename, timesteps=90):
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
                     'data/2D_benchmarks/fixed_1_1/2d_fixed_1_1_all.h5',
                     'data/2D_benchmarks/fixed_1_2/2d_fixed_1_2_all.h5',
                     'data/2D_benchmarks/fixed_1_25/2d_fixed_1_25_all.h5',
                     'data/2D_benchmarks/v032_rl/2d_v032_rl_all.h5',
                     'data/2D_benchmarks/random/2d_random_all.h5'
                     ]
    PC_name_list = ['PC No therapy', 'PC MTD', 'PC AT50', 'PC AT100', 'PC fixed 1.1', 'PC fixed 1.2', 'PC fixed 1.25', 'A RL', 'PC Random']

    PC_dict = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])

    PC_df = pd.DataFrame(PC_dict)

    LV_files_list = ['./Evaluations/LvEnvEval_2d_no_treatment.h5',
                        './Evaluations/LvEnvEval_2d_mtd.h5',
                        './Evaluations/LvEnvEval_2d_at50.h5',
                        './Evaluations/LvEnvEval_2d_at100.h5',
                        './Evaluations/LvEnvEval_2d_fixed_1_1.h5',
                        './Evaluations/LvEnvEval_2d_fixed_1_2.h5',
                        './Evaluations/LvEnvEval_2d_fixed_1_25.h5',
                        './Evaluations/LvEnvEval__2d_rl_29022024_mela_2d_newest.h5',
                        './Evaluations/LvEnvEval_2d_random.h5'

                        ]
    LV_name_list = ['LV No therapy', 'LV MTD', 'LV AT50', 'LV AT100', 'LV fixed 1.1', 'LV fixed 1.2', 'LV fixed 1.25', 'LV A RL', 'LV Random']

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

    return combined_df
#
# df = pd.read_hdf('./Evaluations/PcEnvEval_2d_pc_no_treat_position_test.h5', key=f'run_2')
# df_at100 = pd.read_hdf('./Evaluations/PcEnvEval_2d_pc_at100_position_test.h5', key=f'run_1')
# df_fixed = pd.read_hdf('./Evaluations/PcEnvEval_2d_pc_fixed_1_2_position_test.h5', key=f'run_0')
# df_mtd = pd.read_hdf('./Evaluations/PcEnvEval_2d_pc_mtd_2_position_test.h5', key=f'run_1')
#
# fig, ax = plt.subplots()
# ax.plot(df_at100.index, df_at100['Radius'], label='rad')
# ax.plot(df_at100.index, df_at100['Mutant Position']*df_at100['Radius'], label='mutant')
# ax.fill_between(df_at100.index, max(df_at100['Radius'])+10, max(df_at100['Radius'])+100, where=df_at100['Treatment']==1, color='orange', label='drug')
#
# fig, ax = plt.subplots()
# ax.plot(df_fixed.index, df_fixed['Radius'], label='rad')
# ax.plot(df_fixed.index, df_fixed['Mutant Position']*df_fixed['Radius'], label='mutant')
# ax.fill_between(df_fixed.index, max(df_fixed['Radius'])+10, max(df_fixed['Radius'])+100, where=df_fixed['Treatment']==1, color='orange', label='drug')
# ax.set_title('Fixed 1.2')

df = pd.read_hdf('./Evaluations/LvEnvEval_2d_fixed_1_25.h5', key=f'run_0')
plot(df, 'LV fixed 1.25', scale='linear', truncate=False)

df = pd.read_hdf('./data/2D_benchmarks/fixed_1_25/2d_fixed_1_25_all.h5', key=f'run_0')
plot(df, 'PC fixed 1.25', scale='linear', truncate=False)

df_lv = pd.read_hdf('./Evaluations/LvEnvEval__2d_lv_cobrat_t_5_208032024_cobra_2d_t_5_2.h5', key=f'run_0')
plot(df_lv, 'LV cobrat t_5_2', scale='linear', truncate=False)

df_pc = pd.read_hdf('./Evaluations/PcEnvEval_rl_t_5_115rew08032024_cobra_2d_t_5_2.h5', key=f'run_0')
plot(df_pc, 'PC cobrat t_5_2', scale='linear', truncate=False)

df = pd.read_hdf('./Evaluations/LvEnvEval__2d_lv_cobrat_t_9_208032024_cobra_2d_t_9_2.h5', key=f'run_0')
plot(df, 'LV cobrat t_9_2', scale='linear', truncate=False)

df = pd.read_hdf('./Evaluations/PcEnvEval_rl_t_5_110rew08032024_cobra_2d_t_9_2.h5', key=f'run_0')
plot(df, 'PC cobrat t_9_2', scale='linear', truncate=False)

df = pd.read_hdf('./Evaluations/LvEnvEval__2d_lv_cobrat_t_10_208032024_cobra_2d_t_10_2.h5', key=f'run_0')
plot(df, 'LV cobrat t_10_2', scale='linear', truncate=False)

df = pd.read_hdf('./Evaluations/PcEnvEval_rl_t_10_125rew08032024_cobra_2d_t_10_2.h5', key=f'run_0')
plot(df, 'PC cobrat t_10_2', scale='linear', truncate=False)

combined_df = main()
plt.show()
