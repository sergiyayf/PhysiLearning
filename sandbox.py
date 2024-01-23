import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def get_ttps(filename, timesteps=10):
    ttps = []
    for i in range(1,timesteps+1):
        df = pd.read_hdf(filename+f'_{i}.h5', key='run_0')
        # find the largest index with non-zero Type 0 and Type 1
        nz = df[(df['Type 0'] + df['Type 1'] != 0)]
        if len(nz) > 0:
            ttps.append(nz.index[-1])
        else:
            ttps.append(0)
    return ttps

def main():
    PC_files_list = ['data/3D_benchmarks/at100/PcEnvEvalsim_at100', 'data/3D_benchmarks/mtd/PcEnvEvalsim_mtd',
                     'data/3D_benchmarks/no_therapy/PcEnvEvalsim_no_therapy']
    PC_name_list = ['AT100', 'MTD', 'No therapy']

    PC_dict = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])


    PC_df = pd.DataFrame(PC_dict)
    # combined_df = pd.DataFrame(combined)

    # box plot the distribution with scatter using seaborn
    fig, ax = plt.subplots()
    sns.boxplot(data=PC_df, ax=ax, width = 0.3)
    sns.stripplot(data=PC_df, ax=ax, color='black', jitter=0.2, size=2.5)
    # show mean as well
    ax.scatter(PC_df.mean().index, PC_df.mean(), marker='x', color='red', s=50, label='mean')


df = pd.read_hdf('data/3D_benchmarks/mtd/PcEnvEvalsim_mtd_4.h5', key=f'run_0')
fig, ax = plt.subplots()
ax.plot(df.index, df['Type 0'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 0')
ax.plot(df.index, df['Type 1'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1')
ax.plot(df.index, (df['Type 0'] + df['Type 1'])/(df['Type 0'][0]+df['Type 1'][0]), label='total')
ax.legend()
ax.set_title(f'Mtd')
ax.set_yscale('log')
treat = df['Treatment'].values
# replace 0s that are directly after 1 with 1s
#treat = np.where(treat == 0, np.roll(treat, 1), treat)
ax.fill_between(df.index, 1, 1.250, where=treat==1, color='orange', label='drug',
lw=2)

df = pd.read_hdf('data/3D_benchmarks/at100/PcEnvEvalsim_at100_7.h5', key=f'run_0')
fig, ax = plt.subplots()
ax.plot(df.index, df['Type 0'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 0')
ax.plot(df.index, df['Type 1'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1')
ax.plot(df.index, (df['Type 0'] + df['Type 1'])/(df['Type 0'][0]+df['Type 1'][0]), label='total')
ax.legend()
ax.set_title(f'at100')
ax.set_yscale('log')
treat = df['Treatment'].values
# replace 0s that are directly after 1 with 1s
#treat = np.where(treat == 0, np.roll(treat, 1), treat)
ax.fill_between(df.index, 1, 1.250, where=treat==1, color='orange', label='drug',
lw=2)
#fig.savefig('111.png')
main()
plt.show()
