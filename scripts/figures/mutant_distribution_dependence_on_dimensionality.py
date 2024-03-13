import matplotlib as mpl
mpl.use('TkAgg')
import pandas as pd
from os import path
import numpy as np
import h5py
import seaborn as sns
import matplotlib.pyplot as plt


dir = '/home/saif/Projects/PhysiLearning/data'
hdf5_file_3d = path.join(dir, '3d_full_presims_30_days.h5')
hdf5_file_2d = path.join(dir, 'presims_2d.h5')

def get_n_clones(sims, hdf5_file):
    n_clones = []
    for sim in sims:
        df = pd.read_hdf(hdf5_file, key=f'data/clone_info_sim_{sim}')
        n_clones.append(df['n_clones'].values[0])
    return n_clones

def radius(sims, hdf5_file):
    R = []
    R_std = []
    for sim in sims:
        df = pd.read_hdf(hdf5_file, key=f'data/clone_info_sim_{sim}')
        R.append(df['R'].values[0])
        R_std.append(df['R_std'].values[0])
    return R, R_std

def get_core_shell_radius(sims, hdf5_file):
    R = []
    R_std = []
    for sim in sims:
        df = pd.read_hdf(hdf5_file, key=f'data/clone_info_sim_{sim}')
        R.append(df['R_core_shell'].values[0])
        R_std.append(df['R_core_shell_std'].values[0])
    return R, R_std

def min_clone_to_front_distance(sims, hdf5_file):
    min_clone = []
    for sim in sims:
        df = pd.read_hdf(hdf5_file, key=f'data/clone_info_sim_{sim}')
        min_clone.append(df['min_clone_to_front_distance'].values[0])
    return min_clone


def get_all_clones_min_distance(sims, hdf5_file):
    dist_to_front = []
    for sim in sims:
        with h5py.File(hdf5_file) as f:
            # check if sim exists
            if f.get(f'data/clones/sim_{sim}') is None:
                continue
            else:
                keys = f[f'data/clones/sim_{sim}'].keys()

                for key in keys:
                    df = pd.read_hdf(hdf5_file, key=f'data/clones/sim_{sim}/{key}')
                    dist_to_front.append(df['distance_to_front_cell'].min())


    return dist_to_front


def total_cell_count(sims, hdf5_file):
    total_cells = []
    for sim in sims:
        df = pd.read_hdf(hdf5_file, key=f'data/cells/sim_{sim}')
        total_cells.append(len(df))

    return total_cells


if __name__ == '__main__':

    simulations_3d = range(1, 101)
    simulations_2d = range(1, 1001)
    all_dists_to_front_3d = get_all_clones_min_distance(simulations_3d, hdf5_file_3d)
    all_dists_to_front_2d = get_all_clones_min_distance(simulations_2d, hdf5_file_2d)


    # plot cumulative distribution of distances to front in log log
    fig, ax = plt.subplots()
    # normalize distance to front to 1
    all_dists_to_front_3d = np.array(all_dists_to_front_3d) / np.max(all_dists_to_front_3d)
    inverted_3d = 1-np.array(all_dists_to_front_3d)
    sns.ecdfplot(inverted_3d, ax=ax, label='3D, N=100')
    all_dists_to_front_2d = np.array(all_dists_to_front_2d) / np.max(all_dists_to_front_2d)
    inverted_2d = 1 - np.array(all_dists_to_front_2d)
    sns.ecdfplot(inverted_2d, ax=ax, label='2D, N=1000')

    # plot powerlaw fit
    x = np.linspace(np.min(inverted_2d), np.max(inverted_2d), 100)
    y = x**(2)
    ax.plot(x, y, color='r', label=r'$y = x^2$')
    y = x**(3)
    ax.plot(x, y, color='g', label=r'$y = x^3$')

    ax.set_xlabel('Normalized distance to center')
    ax.set_ylabel('Cumulative probability')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()

    plt.show()

