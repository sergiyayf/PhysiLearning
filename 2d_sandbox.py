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


PC_files_list = [
                 'data/2D_benchmarks/x6/2d_x6_all.h5',
                 'data/2D_benchmarks/n2_t4_l3/2d_n2_t4_l3_all.h5',
                 'data/2D_benchmarks/s2_t5_l3/2d_s2_t5_l3_all.h5',
                 ]
PC_name_list = [
                'PC x6', 'PC n2', 'PC s2',
]

PC_dict = {}
mut_prop_dict = {}
for i in range(len(PC_files_list)):
    PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])
    mut_prop_dict[PC_name_list[i]] = get_mutant_proportions(PC_files_list[i])

# classify the mutant proportions into 3 categories, low, medium and high
classified_mut_prop = {}
for i in range(len(PC_name_list)):
    classified_mut_prop[PC_name_list[i]] = np.zeros(len(mut_prop_dict[PC_name_list[i]]))
    for j in range(len(mut_prop_dict[PC_name_list[i]])):
        if mut_prop_dict[PC_name_list[i]][j] < 0.01 and PC_dict[PC_name_list[i]][j] < 251:
            classified_mut_prop[PC_name_list[i]][j] = 0
        elif mut_prop_dict[PC_name_list[i]][j] < 0.5 and PC_dict[PC_name_list[i]][j] < 251:
            classified_mut_prop[PC_name_list[i]][j] = 1
        elif mut_prop_dict[PC_name_list[i]][j] >= 0.5 and PC_dict[PC_name_list[i]][j] < 251:
            classified_mut_prop[PC_name_list[i]][j] = 2
        else:
            classified_mut_prop[PC_name_list[i]][j] = 3

# pie chart classification

for i in range(len(PC_name_list)):
    fig, ax = plt.subplots()
    ax.pie([np.sum(classified_mut_prop[PC_name_list[i]] == 0),
            np.sum(classified_mut_prop[PC_name_list[i]] == 1),
            np.sum(classified_mut_prop[PC_name_list[i]] == 2),
            np.sum(classified_mut_prop[PC_name_list[i]] == 3)],
           labels=['premature progression', 'resistance rise', 'resistance > 50%', 'resistance contained'], autopct='%1.1f%%', startangle=90)
    ax.set_title(PC_name_list[i])


PC_df = pd.DataFrame(PC_dict)
mut_prop_df = pd.DataFrame(mut_prop_dict)
classified_mut_prop_df = pd.DataFrame(classified_mut_prop)

df_time_melted = PC_df.melt(var_name='Agent', value_name='Time to progression')

df_classified_mut_prop_melted = classified_mut_prop_df.melt(var_name='Agent', value_name='Class')

df_merged = pd.merge(df_time_melted, df_classified_mut_prop_melted, left_index=True, right_index=True)
df_merged = df_merged[['Agent_x', 'Time to progression', 'Class']]

fig, ax = plt.subplots()
#sns.boxplot(x='Agent_x', y='Time to progression', data=df_merged, ax=ax)
sns.swarmplot(x='Agent_x', y='Time to progression', hue='Class', data=df_merged, dodge=True, ax=ax)
# sns.stripplot(x='Agent_x', y='Time to progression', hue='Class', data=df_merged,
#               jitter=0.15, dodge=False, marker='o', alpha=0.7, palette='plasma', ax=ax)
plt.show()

# plot proportions of resistance cells versus TTP
fig, ax = plt.subplots()
ax.scatter(PC_dict['PC x6'], mut_prop_dict['PC x6'], label='PC x6')
# ax.scatter(PC_dict['PC n2'], mut_prop_dict['PC n2'], label='PC n2')
# ax.scatter(PC_dict['PC s2'], mut_prop_dict['PC s2'], label='PC s2')
