import numpy as np
from physicell_tools.get_perifery import front_cells
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import h5py


def read_data(fname, run, time):
    with h5py.File(fname, 'r') as f:
        df = pd.read_hdf(fname, f'run_{run}/time_{time}')
    return df

def calculate_distance_to_front_cell(cells, front):
    for i in range(len(front)):
        current_dist = np.sqrt((cells['position_x']-front[i, 0])**2 + (cells['position_y']-front[i, 1])**2 + (cells['position_z']-front[i, 2])**2)
        if i == 0:
            dist = current_dist
        else:
            dist = np.minimum(dist, current_dist)
    return dist

def calculate_distance_to_front(cell_df: pd.DataFrame, front_cell_positions: np.array):
    """
    Calculate distance to front for all cells in cell_df
    and append to the dataset
    """
    # calculate average radius of cells at the front
    R = np.mean(np.sqrt(front_cell_positions[:, 0]**2 + front_cell_positions[:, 1]**2 + front_cell_positions[:, 2]**2))
    cell_df = cell_df.copy()

    cell_df['distance_to_front_circle'] = (np.sqrt((cell_df['position_x'])**2 + (cell_df['position_y'])**2 + (cell_df['position_z'])**2)).values - R
    cell_df['distance_to_front_cell'] = calculate_distance_to_front_cell(cell_df, front_cell_positions)
    cell_df['distance_to_center'] = np.sqrt((cell_df['position_x'])**2 + (cell_df['position_y'])**2 + (cell_df['position_z'])**2)

    return cell_df

def run_is_at_front(df):
    cell_df = df.rename(columns={'x': 'position_x', 'y': 'position_y', 'z': 'position_z', 'type': 'cell_type'})
    if len(cell_df) > 0:
        cell_df['is_at_front'] = np.zeros_like(cell_df['position_x'])
        cell_df['core_shell'] = np.zeros_like(cell_df['position_x'])
        cell_diameter = 14
        growth_threshold = 1e-5
        # slice through z axis to find cells at front
        z = -500
        while not z > 500:
            cells_in_slice = cell_df[(cell_df['position_z'] > z) & (cell_df['position_z'] < z + cell_diameter)]
            if len(cells_in_slice) == 0:
                z += cell_diameter
                continue
            elif len(cells_in_slice) < 10:
                cell_df.loc[cell_df['position_z'].isin(cells_in_slice['position_z']), 'is_at_front'] = 1
                z += cell_diameter
                continue
            else:
                positions, types = front_cells(cells_in_slice)
                cell_df.loc[cell_df['position_x'].isin(positions[:, 0]), 'is_at_front'] = 1
                z += cell_diameter

        # slice through y axis
        y = -500
        while not y > 500:
            cells_in_slice = cell_df[(cell_df['position_y'] > y) & (cell_df['position_y'] < y + cell_diameter)]
            if len(cells_in_slice) == 0:
                y += cell_diameter
                continue
            elif len(cells_in_slice) < 10:
                cell_df.loc[cell_df['position_y'].isin(cells_in_slice['position_y']), 'is_at_front'] = 1
                y += cell_diameter
                continue
            else:
                positions, types = front_cells(cells_in_slice, clip_plane='xz')
                cell_df.loc[cell_df['position_x'].isin(positions[:, 0]), 'is_at_front'] = 1
                y += cell_diameter

        # repeat for the non-growing cells
        non_growing_cells = cell_df[cell_df['transition_rate'] < growth_threshold]

        z = -500
        while not z > 500:
            cells_in_slice = non_growing_cells[
                (non_growing_cells['position_z'] > z) & (non_growing_cells['position_z'] < z + cell_diameter)]
            if len(cells_in_slice) == 0:
                z += cell_diameter
                continue
            elif len(cells_in_slice) < 10:
                cell_df.loc[cell_df['position_z'].isin(cells_in_slice['position_z']), 'core_shell'] = 1
                z += cell_diameter
                continue
            else:
                positions, types = front_cells(cells_in_slice)
                cell_df.loc[cell_df['position_x'].isin(positions[:, 0]), 'core_shell'] = 1
                z += cell_diameter

        # slice through y axis
        y = -500
        while not y > 500:
            cells_in_slice = non_growing_cells[
                (non_growing_cells['position_y'] > y) & (non_growing_cells['position_y'] < y + cell_diameter)]
            if len(cells_in_slice) == 0:
                y += cell_diameter
                continue
            elif len(cells_in_slice) < 10:
                cell_df.loc[cell_df['position_y'].isin(cells_in_slice['position_y']), 'core_shell'] = 1
                y += cell_diameter
                continue
            else:
                positions, types = front_cells(cells_in_slice, clip_plane='xz')
                cell_df.loc[cell_df['position_x'].isin(positions[:, 0]), 'core_shell'] = 1
                y += cell_diameter

    cells_at_front = cell_df[cell_df['is_at_front'] == 1]
    front_cell_positions = cells_at_front[['position_x', 'position_y', 'position_z']].values
    cell_df = calculate_distance_to_front(cell_df, front_cell_positions)
    return cell_df

def calculate_velocity(df_t_1, df_t_2, time_diff):
    # get only cells with unique IDs
    df_t_1_unique = df_t_1.drop_duplicates(subset='ID')
    df_t_2_unique = df_t_2.drop_duplicates(subset='ID')
    df_t_1 = df_t_1_unique[df_t_1_unique['ID'].isin(df_t_2_unique['ID'])]
    df_t_2 = df_t_2_unique[df_t_2_unique['ID'].isin(df_t_1_unique['ID'])]

    df_t_1['velocity_x'] = (df_t_2['position_x'] - df_t_1['position_x']) / time_diff
    df_t_1['velocity_y'] = (df_t_2['position_y'] - df_t_1['position_y']) / time_diff
    df_t_1['velocity_z'] = (df_t_2['position_z'] - df_t_1['position_z']) / time_diff
    return df_t_1

def calculate_normal_vector(df):
    centroid_x = df['position_x'].mean()
    centroid_y = df['position_y'].mean()
    centroid_z = df['position_z'].mean()
    df['normal_x'] = df['position_x'] - centroid_x
    df['normal_y'] = df['position_y'] - centroid_y
    df['normal_z'] = df['position_z'] - centroid_z
    return df

def calculate_angle(df):
    df['normal_magnitude'] = np.sqrt(df['normal_x']**2 + df['normal_y']**2 + df['normal_z']**2)
    df['velocity_magnitude'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2 + df['velocity_z']**2)
    df['dot_product'] = df['normal_x']*df['velocity_x'] + df['normal_y']*df['velocity_y'] + df['normal_z']*df['velocity_z']
    df['angle'] = np.arccos(df['dot_product'] / (df['normal_magnitude'] * df['velocity_magnitude']))
    return df

def bin_cells(df, bin_size):
    df['bin'] = pd.cut(df['distance_to_front_cell'], bins=np.arange(0, df['distance_to_front_cell'].max() + bin_size, bin_size), include_lowest=True)
    return df

def calculate_average_projections(df):
    df['projection'] = df['dot_product'] / df['normal_magnitude']
    average_projections = df.groupby('bin')['projection'].mean()
    median_projections = df.groupby('bin')['projection'].median()
    mode_projections = df.groupby('bin')['projection'].value_counts()
    std_proj = df.groupby('bin')['projection'].std()
    return average_projections, std_proj, median_projections, mode_projections

def plot_velocity_map(df):
    fig, ax = plt.subplots()
    ax.quiver(df['position_x'], df['position_y'], df['velocity_x'], df['velocity_y'], color='black')
    ax.set_title('2D velocity map')
    plt.show()

def plot_radial_velocity(ax, average_projections, std_proj, bin_size, **kwargs):

    rads = [x*bin_size for x in range(0,len(average_projections))]
    ax.errorbar(rads, average_projections, yerr=std_proj, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    ax.set_xlabel('Distance to front')
    ax.set_ylabel('Normal velocity magnitude')
    ax.set_title('Instant radial velocity')
    plt.show()

if __name__ == '__main__':
    import os
    os.chdir('/home/saif/Projects/PhysiLearning')
    fname = f'./data/3D_benchmarks/p62_pcdl_no_treat/run_1/Evaluations/sim_full_data/pcdl_data_job_10635022_port_0.h5'
    fnames = [f'./data/3D_benchmarks/p62_pcdl_no_treat/run_{i}/Evaluations/sim_full_data/pcdl_data_job_106350{i+21}_port_0.h5' for i in range(1, 11)
              ]
    df = pd.read_hdf(fname, key=f'run_2/time_720')
    cdf = run_is_at_front(df)

    i = 1

    avg_projections = []
    median_projections = []
    mode_projections = []
    std_projs = []
    for fname in fnames:
        for j in range(2, 3):

            df_t_1 = read_data(fname, run=j, time=i * 720)
            df_t_2 = read_data(fname, run=j, time=(i + 1) * 720)
            df_t_1 = run_is_at_front(df_t_1)
            df_t_2 = run_is_at_front(df_t_2)

            df_velocity = calculate_velocity(df_t_1, df_t_2, time_diff=1)
            df_velocity = calculate_normal_vector(df_velocity)
            df_velocity = calculate_angle(df_velocity)
            df_velocity = bin_cells(df_velocity, bin_size=20)
            average_projections, std_proj, med_projections, mode_projection = calculate_average_projections(df_velocity)
            avg_projections.append(average_projections)
            median_projections.append(med_projections)
            mode_projections.append(mode_projection)
            std_projs.append(std_proj)
        # plot_velocity_map(df_velocity)
        # plot_radial_velocity(average_projections, std_proj, bin_size=20)
    # average the projections by same bin
    mean_projections = pd.concat(avg_projections, axis=1).mean(axis=1)
    median_projections = pd.concat(median_projections, axis=1).median(axis=1)

    # calculate the standard deviation by gaussian rule
    std_projs = pd.concat(std_projs, axis=1)
    std_projs = np.sqrt((std_projs ** 2).sum(axis=1))
    rads = [x * 20 for x in range(0, len(mean_projections))]
    rads = np.array(rads)
    fig, ax = plt.subplots()
    plot_radial_velocity(ax, mean_projections, std_projs, bin_size=20)
    #ax.errorbar(rads, median_projections, yerr=std_proj, fmt='d', color='black', ecolor='lightgray', elinewidth=3,
    #            capsize=0)

    # plot horizontal lien at 0
    ax.axhline(0, color='black', lw=1)

    # fit exponential function to the data
    from scipy.optimize import curve_fit

    def exponential(x, a, b, c):
        return a * np.exp(-b * x) + c


    popt, pcov = curve_fit(exponential, rads, mean_projections.values, sigma=std_projs)
    print(popt)
    print(pcov)

    #ax.plot(rads, exponential(rads, *popt), 'r-')
    plt.show()

    # fit linear to the first 5 points
    def linear(x, a, b):
        return a * x + b

    popt, pcov = curve_fit(linear, rads[:4], mean_projections.values[:4], sigma=std_projs[:4])
    print(popt)
    print(pcov)

    popt5, pcov5 = curve_fit(linear, rads[:5], mean_projections.values[:5], sigma=std_projs[:5])
    print(popt5)
    print(pcov5)

    # fit quadratic
    qpopt, qpcov = curve_fit(lambda x, a, b: b * (a - x) ** 2, rads[:5], mean_projections.values[:5], sigma=std_projs[:5])
    def trunc_linear(x, a, b):
        return (a * x + b)*np.heaviside(a*x+b, 1)

    ax.plot(rads, trunc_linear(rads, *popt), 'g-', label='fit 4 points')
    ax.plot(rads, trunc_linear(rads, *popt5), 'b-', label='fit 5 points')
    ax.plot(rads, qpopt[1]*(qpopt[0]-rads)**2, 'r-', label='quad fit')
    ax.legend()
    plt.show()

