import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def get_ttps(filename, timesteps=90):
    ttps = []
    for i in range(timesteps):
        df = pd.read_hdf(filename, key=f'run_{i}')
        # find the largest index with non-zero Type 0 and Type 1
        nz = df[(df['Type 0'] + df['Type 1'] != 0)]
        if len(nz) > 0:
            ttps.append(nz.index[-1])
        else:
            ttps.append(0)
    return ttps

def main():
    PC_files_list = ['data/3D_benchmarks/no_treatment/no_treatment_all.h5',
                     'data/3D_benchmarks/mtd/mtd_all.h5',
                     'data/3D_benchmarks/at100/at100_all.h5',
                     'data/3D_benchmarks/random/random_all.h5'
                     ]
    PC_name_list = ['PC No therapy', 'PC MTD', 'PC AT100', 'PC Random']

    PC_dict = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])

    PC_df = pd.DataFrame(PC_dict)

    LV_files_list = ['./Evaluations/LvEnvEvalno_treatment_check_params_LV.h5',
                        './Evaluations/LvEnvEvalmtd_check_params_LV.h5',
                        './Evaluations/LvEnvEvalat100_check_params_LV.h5',
                        './Evaluations/LvEnvEvalrandom_check_params_LV.h5'
                        ]
    LV_name_list = ['LV No therapy', 'LV MTD', 'LV AT100', 'LV Random']

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



df = pd.read_hdf('data/3D_benchmarks/at100/at100_all.h5', key=f'run_0')
fig, ax = plt.subplots()
ax.plot(df.index, df['Type 0'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 0')
ax.plot(df.index, df['Type 1'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1')
ax.plot(df.index, (df['Type 0'] + df['Type 1'])/(df['Type 0'][0]+df['Type 1'][0]), label='total')
ax.legend()
ax.set_title(f'at100 r0')
ax.set_yscale('log')
treat = df['Treatment'].values
# replace 0s that are directly after 1 with 1s
#treat = np.where(treat == 0, np.roll(treat, 1), treat)
ax.fill_between(df.index, 1, 1.250, where=treat==1, color='orange', label='drug',
lw=2)

df = pd.read_hdf('Evaluations/LvEnvEvalat100_check_params_LV.h5', key=f'run_0')
fig, ax = plt.subplots()
ax.plot(df.index, df['Type 0'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 0')
ax.plot(df.index, df['Type 1'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1')
ax.plot(df.index, (df['Type 0'] + df['Type 1'])/(df['Type 0'][0]+df['Type 1'][0]), label='total')
ax.legend()
ax.set_title(f'at100 r0')
ax.set_yscale('log')
treat = df['Treatment'].values
# replace 0s that are directly after 1 with 1s
#treat = np.where(treat == 0, np.roll(treat, 1), treat)
ax.fill_between(df.index, 1, 1.250, where=treat==1, color='orange', label='drug',
lw=2)


main()
plt.show()
