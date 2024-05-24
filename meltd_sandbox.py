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
    time = df.index[::2]/2
    ax.plot(time, df['Type 0'].values[::2]/(df['Type 0'][0]+df['Type 1'][0]), label='Type 0')
    ax.plot(time, df['Type 1'].values[::2]/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1')
    ax.plot(time, (df['Type 0'][::2] + df['Type 1'][::2])/(df['Type 0'][0]+df['Type 1'][0]), label='total')
    ax.legend()
    ax.set_title(title)
    ax.set_yscale(scale)
    treat = df['Treatment'].values[::2]
    # replace 0s that are directly after 1 with 1s
    treat = np.where(treat == 0, np.roll(treat, -1), treat)
    ax.fill_between(time, 1.0, 1.25, where=treat == 1, color='orange', label='drug',
                       lw=2)
    return ax

def get_ttps(filename, timesteps=100):
    ttps = []
    for i in range(timesteps):
        df = pd.read_hdf(filename, key=f'run_{i}')
        # find the largest index with non-zero Type 0 and Type 1
        initial_size = df['Type 0'][0] + df['Type 1'][0]
        nz = df[((df['Type 0'] + df['Type 1'])/initial_size > 3.0)]
        if len(nz) > 0:
            # append index when type 0 + type 1 is larger than 1.5
            ttps.append(nz.index[0]/2)
        else:
            ttps.append(len(df)/2)
    return ttps

def main():

    LV_files_list = ['./Evaluations/MeltdEnvEval__test_meltd_env_no_treatment.h5',
                        './Evaluations/MeltdEnvEval__test_meltd_env_mtd.h5',
                        './Evaluations/MeltdEnvEval__test_meltd_env_fixed_1_5.h5',
                        './Evaluations/MeltdEnvEval_stupid_agent2305_2d_meltd_noise_agent_t3.h5',
                        './Evaluations/MeltdEnvEval__cobra_rew_4_t14.h5',
                        './Evaluations/MeltdEnvEval__cobra_rew_0_t17.h5',
                        './Evaluations/MeltdEnvEval__rand.h5'
                     ]
    LV_name_list = ['No treatment', 'MTD', 'Fixed 1_5', 'Stupid agent', 'Cobra r4 agent 14', 'C 17', 'Random']

    LV_dict = {}
    for i in range(len(LV_files_list)):
        LV_dict[LV_name_list[i]] = get_ttps(LV_files_list[i])

    LV_df = pd.DataFrame(LV_dict)

    # box plot the distribution with scatter using seaborn
    fig, ax = plt.subplots()
    sns.boxplot(data=LV_df, ax=ax, width = 0.3)
    sns.stripplot(data=LV_df, ax=ax, color='black', jitter=0.2, size=2.5)
    # show mean as well
    ax.scatter(LV_df.mean().index, LV_df.mean(), marker='x', color='red', s=50, label='mean')



df = pd.read_hdf('./Evaluations/MeltdEnvEval__test_meltd_env_no_treatment.h5', key=f'run_0')
plot(df, 'No treatment', scale='linear', truncate=False)

df = pd.read_hdf('./Evaluations/MeltdEnvEval__test_meltd_env_mtd.h5', key=f'run_0')
plot(df, 'Meltd MTD', scale='linear', truncate=False)

for k in range(2):

    df = pd.read_hdf('./Evaluations/MeltdEnvEval_stupid_agent2305_2d_meltd_noise_agent_t3.h5', key=f'run_{k}')
    plot(df, f'Meltd mela Agent run_{k}', scale='linear', truncate=False)

    df = pd.read_hdf('./Evaluations/MeltdEnvEval__cobra_rew_4_t14.h5', key=f'run_{k}')
    plot(df, f'Meltd Cobra 14 run_{k}', scale='linear', truncate=False)

    df = pd.read_hdf('./Evaluations/MeltdEnvEval__cobra_rew_0_t17.h5', key=f'run_{k}')
    plot(df, f'Meltd Cobra 17 run_{k}', scale='linear', truncate=False)

main()
plt.show()
