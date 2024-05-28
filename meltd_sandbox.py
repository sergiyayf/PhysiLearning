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
    # plot radius and mutant position as well
    # ax.plot(time, df['Radius'].values[::2]/df['Radius'].values[0], label='Radius', color='black')
    # ax.plot(time, df['Mutant Position'].values[::2], label='Mutant', color='red')
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
        df = df[::4]
        # find the largest index with non-zero Type 0 and Type 1
        initial_size = df['Type 0'][0] + df['Type 1'][0]
        nz = df[((df['Type 0'] + df['Type 1'])/initial_size > 3.0)]
        if len(nz) > 0:
            # append index when type 0 + type 1 is larger than 1.5
            ttps.append(nz.index[0]/2)
        else:
            ttps.append(len(df)*2)
    return ttps

def main():

    LV_files_list = ['./Evaluations/MeltdEnvEval__new_model_no_treatment.h5',
                        './Evaluations/MeltdEnvEval__new_model_mtd.h5',
                      './Evaluations/MeltdEnvEval__new_model_at100.h5',
                        './Evaluations/MeltdEnvEval__new_model_fixed_1_5.h5',
                        './Evaluations/MeltdEnvEval_24ct1_final2.h5',
                        './Evaluations/MeltdEnvEval_24ct11.h5'

                     ]
    LV_name_list = ['No treatment', 'MTD', 'at100', 'Fixed 1_5', 'agent ct1 final2', 'agent ct11']

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
    # horizontal line at MTD median
    ax.axhline(y=LV_df['MTD'].median(), color='blue', linestyle='--', label='MTD median')


for i in range(6):

    # df = pd.read_hdf('./Evaluations/MeltdEnvEval__new_model_mtd.h5', key=f'run_{i}')
    # plot(df, 'MTD', scale='linear', truncate=False)
    # df = pd.read_hdf('./Evaluations/MeltdEnvEval__new_model_at100.h5', key=f'run_{i}')
    # plot(df, 'at100', scale='linear', truncate=False)
    # df = pd.read_hdf('./Evaluations/MeltdEnvEval__new_model_fixed_1_5.h5', key=f'run_{i}')
    # plot(df, 'Fixed 1_5', scale='linear', truncate=False)
    df = pd.read_hdf('./Evaluations/MeltdEnvEval_24ct1_final2.h5', key=f'run_{i}')
    plot(df, 'Agent', scale='linear', truncate=False)
    df = pd.read_hdf('./Evaluations/MeltdEnvEval_24ct11.h5', key=f'run_{i}')
    plot(df, 'Agent 11', scale='linear', truncate=False)

main()
plt.show()
