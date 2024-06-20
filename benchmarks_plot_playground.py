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

def plot_mutant_proportions(filename, timesteps=100):
    mutant_proportions = get_mutant_proportions(filename, timesteps)
    fig, ax = plt.subplots()
    ax.plot(range(timesteps), mutant_proportions)
    ax.set_title('Mutant proportions over time')
    return ax


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

combined_df = pd.concat([PC_df, LV_df], axis=1)
# plot LV ttps against PC ttps
fig, ax = plt.subplots()
for i in range(len(PC_df.columns)):
    #ax.scatter(LV_df[LV_df.columns[i]], PC_df[PC_df.columns[i]], label=PC_df.columns[i])
    sns.stripplot(data=combined_df, x=LV_name_list[i], y=PC_name_list[i], jitter=2, size=4,
                  ax=ax, label=PC_name_list[i], native_scale=True)

# plot diagonal
ax.plot([0, 150], [0, 150], 'k--')
ax.set_xlabel('LV')
ax.set_ylabel('PC')
ax.legend()


plt.show()
