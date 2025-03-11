from hmac import digest_size

import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from SI.auxiliary import run_is_at_front, read_data
ex = {'font.size': 6,
          'font.weight': 'normal',
          'pdf.fonttype': 42,
          'font.family': 'sans-serif',
          'font.sans-serif': 'Arial',
          'errorbar.capsize': 2,
          }
plt.rcParams.update(ex)
def plot_growth_rate_distribution(df):
    # distribution of growth rates as a function of distance to front, mean and quantiles
    fig, ax = plt.subplots()

    # bin data by distance to front cell, calculate mean and std
    bin_size = 10
    df['bin'] = pd.cut(df['distance_to_front_cell'], bins=np.arange(0, df['distance_to_front_cell'].max() + bin_size, bin_size), include_lowest=True)
    df_grouped = df.groupby('bin')
    mean = df_grouped['transition_rate'].mean()
    std = df_grouped['transition_rate'].std()
    dists = [x*bin_size for x in range(len(mean))]
    ax.errorbar(dists, mean, yerr=std, fmt='o', label='Mean and std')

    ax.legend()
    ax.set_xlabel('Distance to front cell')
    ax.set_ylabel('Transition rate')
    plt.show()

    return dists, mean, std



if __name__ == '__main__':
    # set pwd
    import os
    os.chdir('/home/saif/Projects/PhysiLearning')
    # df = read_data('./data/position_physilearning/transition_rate_save_run_1/Evaluations/sim_full_data/pcdl_data_job_7839832_port_0.h5', 2, 1*720)
    for t in range(1, 7):
        time = 720*t
        for i in range(2,5):
            df = read_data(f'/media/saif/1A6A95E932FFC943/nc/run_2025/Evaluations/sim_full_data/pcdl_data_job_15040854_port_0.h5', i, time)
            if i == 2:
                df2 = run_is_at_front(df)
                df_all = df2
            else:
                df2 = run_is_at_front(df)
                df_all = pd.concat([df_all, df2])
    df = read_data(f'/media/saif/1A6A95E932FFC943/nc/run_2025/Evaluations/sim_full_data/pcdl_data_job_15040854_port_0.h5', 6, 2880)
    #df = pd.read_pickle('for_velocity_plotting_10_nc_runs_df.pkl')
    df_all['transition_rate'] = df_all['transition_rate']*720
    # plot cell positions, color map by growth rate
    fig, ax = plt.subplots(figsize=(150/72,150/72), constrained_layout=True)
    sns.scatterplot(x='x', y='y', data=df, hue='transition_rate', ax=ax, s=3)
    # plot colorbar instead of legend
    ax.legend([],[], frameon=False)
    # # get colorbar
    cm = sns.cubehelix_palette(as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=df['transition_rate'].min(), vmax=df['transition_rate'].max()))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    # set colorbar ticks format to scientific notation
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()
    ax.axis('equal')
    fig.savefig('./scripts/figures/plots/Figure_3_d_growth_rate_distribution.pdf', transparent = True)

    plt.show()


