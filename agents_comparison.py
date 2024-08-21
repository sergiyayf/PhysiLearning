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

for model in ['l', 'n', 's']:
    fig, ax = plt.subplots()
    combined_df = pd.DataFrame()
    for t in range(1,6):
        PC_files_list = [
            f'./Evaluations/saved_paper_2d_evals/LvEnvEval__20240425_agents_comparison_det_LV_2204_{model}3_t{t}_l2.h5',
            f'./Evaluations/saved_paper_2d_evals/LvEnvEval__20240425_agents_comparison_noise_LV_2204_{model}3_t{t}_l2.h5',
            f'./Evaluations/saved_paper_2d_evals/SLvEnvEval__20240425_agents_comparison_noise_LV_2204_{model}3_t{t}_l2.h5',
            f'./data/2D_benchmarks/agent_{model}3/t{t}/2d_{model}3_t{t}_run_all.h5',
                         ]
        PC_name_list = [f'det {model} t{t}', f'noise {model} t{t}', f'q-s {model} t{t}', f'PC {model} t{t}']
        PC_dict = {}
        for i in range(len(PC_files_list)):
            PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])

        # plot LV ttps against PC ttps
        combined_df = pd.concat([combined_df, pd.DataFrame(PC_dict)], axis=1)

        # box plot the distribution with scatter using seaborn

    sns.boxplot(data=combined_df, ax=ax, width = 0.3)
    sns.stripplot(data=combined_df, ax=ax, color='black', jitter=0.2, size=2.5)
    # show mean as well
    ax.scatter(combined_df.mean().index, combined_df.mean(), marker='x', color='red', s=50, label='mean')
    ax.set_title(f'{model} t{t}')
plt.show()


# plot lt5 and lt2 trajectories
df = pd.read_hdf('./data/2D_benchmarks/agent_l3/t5/2d_l3_t5_run_all.h5', key='run_10')
ax = plot(df, 'det l3 t5 l2')

df = pd.read_hdf('./data/2D_benchmarks/agent_l3/t2/2d_l3_t2_run_all.h5', key='run_10')
ax = plot(df, 'det l3 t2 l2')

for i in range(1,6):
    df = pd.read_hdf(f'./Evaluations/saved_paper_2d_evals/LvEnvEval__20240425_agents_comparison_det_LV_2204_l3_t{i}_l2.h5', key='run_10')
    ax = plot(df, f'det l3 t{i} l2')