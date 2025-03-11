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
    #os.chdir('/media/saif/1A6A95E932FFC943/09012025_onehalf_train_6_physicells/full_run_1/Evaluations')
    position = []
    is_at_front = []
    evaluation_df = pd.read_hdf('/media/saif/1A6A95E932FFC943/14022025_run_of_0602_lv_evaluations_paper/run_2_9.h5', key='run_0')
    end_t = 67
    for t in range(0, end_t):
        time = 720*t

        df = read_data(f'/media/saif/1A6A95E932FFC943/14022025_run_of_0602_lv_evaluations_paper/movies_9/pcdl_2.h5', 2, time)
        df2 = run_is_at_front(df)

        # resistat cell with largest ditance to center
        res = df2[df2['cell_type'] == 1]
        res = res[res['distance_to_front_cell'] == res['distance_to_front_cell'].min()]
        position.append(res['distance_to_front_cell'].values[0])
        is_at_front.append(res['is_at_front'].values[0])


    # plot position over time
    fig, ax = plt.subplots(figsize=(150/72, 150/72), constrained_layout=True)
    colors = ['red' if x == 1 else 'blue' for x in is_at_front]
    ax.plot(np.arange(0, end_t), position, c='b', marker='o')
    ax.set_xlabel('Time')

    treats = evaluation_df['Treatment'][0:end_t*2].values
    x = np.arange(0, end_t*2)/2
    ax.fill_between(x, 0, 250, where=treats == 1, color='gray', alpha=0.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Resistance distance to front')

    fig.savefig('/home/saif/Projects/PhysiLearning/scripts/figures/plots/Figure_3_ratchet.pdf', transparent=True)
    # with h5py.File(f'./sim_full_data/pcdl_data_job_14948503_port_0.h5', 'r') as f:
    #     runs = list(f.keys())
    #     times = list(f['run_3'].keys())
    #plt.show()


