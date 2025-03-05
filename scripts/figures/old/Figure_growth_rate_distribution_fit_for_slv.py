from hmac import digest_size

import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from SI.auxiliary import run_is_at_front, read_data

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
    os.chdir('/media/saif/1A6A95E932FFC943/nc/run_2025/Evaluations/sim_full_data')
    # df = read_data('./data/position_physilearning/transition_rate_save_run_1/Evaluations/sim_full_data/pcdl_data_job_7839832_port_0.h5', 2, 1*720)
    for t in range(0, 7):
        time = 720*t
        for i in range(2,52):
            df = read_data(f'./pcdl_data_job_15040854_port_0.h5', i, time)
            if i == 2:
                df2 = run_is_at_front(df)
                df_all = df2
            else:
                df2 = run_is_at_front(df)
                df_all = pd.concat([df_all, df2])
    #df = read_data(f'./data/29112024_2d_manuals/nc/sim_full_data/pcdl_data_job_13887080_port_0.h5', 6, 2880)
    #df = pd.read_pickle('for_velocity_plotting_10_nc_runs_df.pkl')
    df_all['transition_rate'] = df_all['transition_rate']*360
    # plot cell positions, color map by growth rate
    # fig, ax = plt.subplots(figsize=(6,5))
    # sns.scatterplot(x='x', y='y', data=df, hue='transition_rate', ax=ax, s=12)
    # # plot colorbar instead of legend
    # ax.legend([],[], frameon=False)
    # # get colorbar
    # cm = sns.cubehelix_palette(as_cmap=True)
    # sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=df['transition_rate'].min(), vmax=df['transition_rate'].max()))
    # sm.set_array([])
    # fig.colorbar(sm, ax=ax)
    # fig.savefig('./plots/growth_rate_position_dependence_poster.pdf', transparent = True)
    #df2 = run_is_at_front(df)
    df2 = df_all
    dists, mean, std = plot_growth_rate_distribution(df2)
    dists, mean, std = np.array(dists), np.array(mean), np.array(std)

    # fit line to the points that greater than zero
    def linear(x, a, b):
        return a*x + b

    from scipy.optimize import curve_fit
    # select only the points that are greater than zero
    to_fit_dists = dists[mean-std > 0]
    to_fit_mean = mean[mean-std > 0]
    popt, pcov = curve_fit(linear, to_fit_dists, to_fit_mean)
    # print(popt)

    def trunc_linear(x, a, b):
        return (a * x + b) * np.heaviside(a * x + b, 1)


    def exp (x, a, b):
        return a*np.exp(-b*x)

    popt_exp, pcov_exp = curve_fit(exp, dists, mean)
    # print('Exp: ', popt_exp)

    def quadratic(x, a, b):
        return b*(a-x)**2

    def trunc_quadratic(x, a, b):
        return (b*(a-x)**2) * np.heaviside((a-x), 1)-0.001

    popt_quad, pcov_quad = curve_fit(quadratic, dists[0:20], mean[0:20])
    print('Quad fit: ', popt_quad)

    fig, ax = plt.subplots()
    ax.errorbar(dists, mean, yerr=std, fmt='o', label='Mean and std')
    # ax.plot(dists, trunc_linear(dists, *popt), label='Fit')
    # ax.plot(dists, exp(dists, *popt_exp), label='Exp fit')
    ax.plot(dists, trunc_quadratic(dists, *popt_quad), label='Quadratic fit')
    ax.legend()
    ax.set_xlabel('Distance to front cell')
    ax.set_ylabel('Transition rate')
    plt.show()

    # Res: 2.20265e+02, 2.3843e-06

