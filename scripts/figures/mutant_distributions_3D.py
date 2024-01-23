import matplotlib as mpl
mpl.use('TkAgg')
import pandas as pd
from os import path
import numpy as np
import h5py
import seaborn as sns
import matplotlib.pyplot as plt


dir = '/home/saif/Projects/PhysiLearning/data'
hdf5_file = path.join(dir, 'presims_3d.h5')


def get_n_clones(sims):
    n_clones = []
    for sim in sims:
        df = pd.read_hdf(hdf5_file, key=f'data/clone_info_sim_{sim}')
        n_clones.append(df['n_clones'].values[0])
    return n_clones

def radius(sims):
    R = []
    R_std = []
    for sim in sims:
        df = pd.read_hdf(hdf5_file, key=f'data/clone_info_sim_{sim}')
        R.append(df['R'].values[0])
        R_std.append(df['R_std'].values[0])
    return R, R_std

def get_core_shell_radius(sims):
    R = []
    R_std = []
    for sim in sims:
        df = pd.read_hdf(hdf5_file, key=f'data/clone_info_sim_{sim}')
        R.append(df['R_core_shell'].values[0])
        R_std.append(df['R_core_shell_std'].values[0])
    return R, R_std

def min_clone_to_front_distance(sims):
    min_clone = []
    for sim in sims:
        df = pd.read_hdf(hdf5_file, key=f'data/clone_info_sim_{sim}')
        min_clone.append(df['min_clone_to_front_distance'].values[0])
    return min_clone


def get_all_clones_min_distance(sims):
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

if __name__ == '__main__':
    n_clones= get_n_clones(range(1, 101))
    R, R_std = radius(range(1, 101))
    R_core_shell, R_core_shell_std = get_core_shell_radius(range(1, 101))
    min_clone = min_clone_to_front_distance(range(1, 101))
    all_dists_to_front = get_all_clones_min_distance(range(1, 101))

    print('Median distance to front: ', np.median(all_dists_to_front))

    # plot distribution of distances to front
    sns.histplot(all_dists_to_front)

    # plot number of clones distribution
    plt.figure()
    sns.histplot(n_clones)

    # plot radius distribution
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.histplot(R, ax=ax)
    ax.set_xlabel('Radius of sphere')
    ax.set_ylabel('Count')

    # plot core shell radius distribution
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.histplot(R_core_shell, ax=ax)
    ax.set_xlabel('Radius of core shell')
    ax.set_ylabel('Count')

    # plot growth layer distribution
    fig = plt.figure()
    ax = fig.add_subplot(111)
    growth_layer = np.array(R) - np.array(R_core_shell)
    sns.histplot(growth_layer, ax=ax)
    ax.set_xlabel('Radius of growth layer')
    ax.set_ylabel('Count')


    plt.show()

