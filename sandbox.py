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
        nz = df[((df['Type 0'] + df['Type 1'])/initial_size > 1.4)]
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
                     #'data/3D_benchmarks/random/random_all.h5'
                     ]
    PC_name_list = ['PC No therapy', 'PC MTD', 'PC AT100']

    PC_dict = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])

    PC_df = pd.DataFrame(PC_dict)

    LV_files_list = ['./Evaluations/temp/LvEnvEvalno_treatment_1_5.h5',
                        './Evaluations/LvEnvEval__high_low_1_10_0_90.h5',
                        './Evaluations/LvEnvEval__high_low_1_20_0_80.h5',
                        './Evaluations/SLvEnvEval__high_low_1_10_0_90.h5',
                        './Evaluations/SLvEnvEval__high_low_1_20_0_80.h5',
                        #'./Evaluations/LvEnvEvalrandom_1_5.h5'
                        ]
    LV_name_list = ['LV No therapy', 'LV s', 'LV w', 'Slv shall', 'Slv wide']

    LV_dict = {}
    for i in range(len(LV_files_list)):
        LV_dict[LV_name_list[i]] = get_ttps(LV_files_list[i])

    LV_df = pd.DataFrame(LV_dict)

    # combine the two dataframes
    combined = {}
    for i in range(len(PC_name_list)):
        combined[PC_name_list[i]] = PC_df[PC_name_list[i]]
        # combined[LV_name_list[i]] = LV_df[LV_name_list[i]]
    # combined_df = pd.DataFrame(combined)
    combined_df = LV_df
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


# for i in range(1,10):
#     df = pd.read_hdf(f'./data/temp/agents_updated_progression_def/2_test_{i}.h5', key=f'run_0')
#     plot(df, f'LV no treatment 1.{i}', scale='linear', truncate=False)
#     average = df['Type 0'].values[:100]
#     print(f'1.{i} average: {average.mean()}')
#     ttp = get_ttps(f'./data/temp/agents_updated_progression_def/test_{i}.h5')
#     print(f'1.{i} ttp: {np.mean(ttp)}')

# for i in range(1,10):
#     df = pd.read_hdf(f'./data/temp/deep.h5', key=f'run_{i}')
#     plot(df, f'deep {i}', scale='linear', truncate=False)
#     df = pd.read_hdf(f'./data/temp/shallow.h5', key=f'run_{i}')
#     plot(df, f'shallow {i}', scale='linear', truncate=False)

meltd_df = pd.read_hdf('./Evaluations/meltd/MeltdEnvEval_1006_2d_meltd_l2.h5', key=f'run_0')
plot(meltd_df, 'Meltd stupid agent', scale='linear', truncate=False)
# main()
plt.show()
