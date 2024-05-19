import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from auxiliary import run_3d_is_at_front, read_data

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

def cart_to_sphere(df):
    df['rho'] = np.sqrt(df['x']**2 + df['y']**2)
    df['phi'] = np.arctan2(df['y'], df['x'])
    df['theta'] = np.arccos(df['z'] / df['rho'])
    return df

if __name__ == '__main__':
    # set pwd
    import os
    os.chdir('/home/saif/Projects/PhysiLearning')
    df = read_data('./data/3D_benchmarks/p62_pcdl_no_treat/run_1/Evaluations/sim_full_data/pcdl_data_job_10635022_port_0.h5', 2, 1*720)
    df['transition_rate'] = df['transition_rate']*720
    # make a z slice for cells wiht -10< z < 10
    # df = df[(df['z'] > -10) & (df['z'] < 10)]
    # check if there is transition rate grater than 1, if so remove
    df = df[df['transition_rate'] < 1]
    # plot cell positions, color map by growth rate
    fig, ax = plt.subplots()
    sns.scatterplot(x='x', y='y', data=df, hue='transition_rate', ax=ax)

    df_sp = cart_to_sphere(df)
    fig, ax = plt.subplots()
    sns.scatterplot(x='rho', y='theta', data=df_sp, hue='transition_rate', ax=ax)

    df_sp = cart_to_sphere(df)
    fig, ax = plt.subplots()
    sns.scatterplot(x='rho', y='phi', data=df_sp, hue='transition_rate', ax=ax)
    df2 = run_3d_is_at_front(df)

    # 3D scatter cells, color differently if they are at the front, alpha = 0.5
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df2['position_x'], df2['position_y'], df2['position_z'], c='r', marker='o', alpha=0.5)
    ax.scatter(df2['position_x'][df2['is_at_front'] == 1], df2['position_y'][df2['is_at_front'] == 1], df2['position_z'][df2['is_at_front'] == 1], c='b', marker='o', alpha=0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['x'], df['y'], df['z'], c='r', marker='o', alpha=0.5)


    dists, mean, std = plot_growth_rate_distribution(df2)
    dists, mean, std = np.array(dists), np.array(mean), np.array(std)

    # fit line to the points that greater than zero
    def linear(x, a, b):
        return a*x + b


    from scipy.optimize import curve_fit

    # select only the points that are greater than zero
    to_fit_dists = dists[mean - std > 0]
    to_fit_mean = mean[mean - std > 0]
    popt, pcov = curve_fit(linear, to_fit_dists, to_fit_mean)
    print(popt)
    # fit quadratic
    qpopt, qpcov = curve_fit(lambda x, a, b: b * (a - x) ** 2, to_fit_dists, to_fit_mean)
    print('Parabolic: ', qpopt)


    def trunc_linear(x, a, b):
        return (a * x + b) * np.heaviside(a * x + b, 1)


    def trunc_quad(x, a, b):
        # split around x = a
        return (b * (a - x) ** 2) * np.heaviside(a - x, 1)


    def exp(x, a, b):
        return a * np.exp(-b * x)


    popt_exp, pcov_exp = curve_fit(exp, dists, mean)
    print(popt_exp)

    fig, ax = plt.subplots()
    ax.errorbar(dists, mean, yerr=std, fmt='o', label='Mean and std')
    ax.plot(dists, trunc_linear(dists, *popt), label='Fit')
    ax.plot(dists, exp(dists, *popt_exp), label='Exp fit')
    ax.plot(dists, trunc_quad(dists, *qpopt), label='Quad fit')
    ax.legend()
    ax.set_xlabel('Distance to front cell')
    ax.set_ylabel('Transition rate')
    plt.show()


