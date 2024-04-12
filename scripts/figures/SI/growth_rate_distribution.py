import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from auxiliary import run_is_at_front, read_data

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
    df = read_data('./data/position_physilearning/transition_rate_save_run_1/Evaluations/sim_full_data/pcdl_data_job_7839832_port_0.h5', 2, 1*720)
    df['transition_rate'] = df['transition_rate']*720
    # plot cell positions, color map by growth rate
    fig, ax = plt.subplots()
    sns.scatterplot(x='x', y='y', data=df, hue='transition_rate', ax=ax)
    df2 = run_is_at_front(df)
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
    print(popt)

    def trunc_linear(x, a, b):
        return (a * x + b) * np.heaviside(a * x + b, 1)


    def exp (x, a, b):
        return a*np.exp(-b*x)

    popt_exp, pcov_exp = curve_fit(exp, dists, mean)
    print(popt_exp)

    fig, ax = plt.subplots()
    ax.errorbar(dists, mean, yerr=std, fmt='o', label='Mean and std')
    ax.plot(dists, trunc_linear(dists, *popt), label='Fit')
    ax.plot(dists, exp(dists, *popt_exp), label='Exp fit')
    ax.legend()
    ax.set_xlabel('Distance to front cell')
    ax.set_ylabel('Transition rate')
    plt.show()


