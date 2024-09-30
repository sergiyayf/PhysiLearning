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

    skip = 2
    normalize = 2
    sus = np.array(df['Type 0'].values )#/ (df['Type 0'][normalize] + df['Type 1'][normalize]))
    res = np.array(df['Type 1'].values )#/ (df['Type 0'][normalize] + df['Type 1'][normalize]))
    # sus = np.array(df['Type 0'].values)*70
    # res = np.array(df['Type 1'].values)*70
    tot = sus + res
    time = df.index / 2

    ax.plot(time[::skip], sus[::skip], color='g')
    ax.plot(time[::skip], res[::skip], color='r')
    ax.plot(time[::skip], tot[::skip], color='black', label='LV')

    ax.legend()
    ax.set_title(title)
    ax.set_yscale(scale)
    ax.hlines(1.0, 0, time[-1], color='k', linestyle='--')
    treat = df['Treatment'].values
    # replace 0s that are directly after 1 with 1s
    treat = np.where(treat == 0, np.roll(treat, -1), treat)
    ax.fill_between(time, 0, 200, where=treat==1, color='orange', label='drug',
    lw=2)

    b = np.array([6322, 7215, 8159, 8246, 6563, 5068, 5393, 6203, 6695, 8155, 10244, 12520, 13013])
    d = np.array([6589, 7531, 8595, 8662, 7559, 6745, 6539, 7280, 7845, 8666, 9837, 11615, 11724])
    c = np.array([5845, 6865, 7957, 8014, 6781, 5245, 5274, 6242, 6550, 7790, 9647, 11916, 12452])
    #ax.scatter(range(len(b)), np.array(b), color='green', label='B7')
    #ax.scatter(range(len(c)), np.array(c), color='blue', label='C7')
    #ax.scatter(range(len(d)), np.array(d), color='red', label='D7')
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized cell count')
    ax.legend()
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

df_mtd = pd.read_hdf('./Evaluations/LvEnvEval__5000_2_mtd.h5', key='run_0')
plot(df_mtd, 'Lv 5000 2 mtd ', scale='linear', truncate=False)

df_mtd = pd.read_hdf('./Evaluations/LvEnvEval__8000_1_mtd.h5', key='run_0')
plot(df_mtd, 'Lv 8000 1 mtd ', scale='linear', truncate=False)

df_mtd = pd.read_hdf('./Evaluations/LvEnvEval__5000_2_eat.h5', key='run_0')
plot(df_mtd, 'Lv 5000 2 eat ', scale='linear', truncate=False)

df_mtd = pd.read_hdf('./Evaluations/LvEnvEval__8000_1_eat.h5', key='run_0')
plot(df_mtd, 'Lv 8000 1 eat ', scale='linear', truncate=False)

df_metld = pd.read_hdf('./Evaluations/MeltdEnvEval__test_meltd_at100.h5', key='run_0')
plot(df_metld, 'Meltd at100', scale='linear', truncate=False)

df_m = pd.read_hdf('./Evaluations/meltd/MeltdEnvEval_1006_2d_meltd_l2.h5', key='run_0')
plot(df_m, 'Meltd 1006 2d l2', scale='linear', truncate=False)
plt.show()
