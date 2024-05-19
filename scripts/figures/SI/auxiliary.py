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

        cells_in_slice = cell_df
        if len(cells_in_slice) == 0:
            pass
        elif len(cells_in_slice) < 10:
            cell_df.loc[cell_df['position_z'].isin(cells_in_slice['position_z']), 'is_at_front'] = 1

        else:
            positions, types = front_cells(cells_in_slice)
            cell_df.loc[cell_df['position_x'].isin(positions[:, 0]), 'is_at_front'] = 1

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
    return df_t_1

def calculate_normal_vector(df):
    centroid_x = df['position_x'].mean()
    centroid_y = df['position_y'].mean()
    df['normal_x'] = df['position_x'] - centroid_x
    df['normal_y'] = df['position_y'] - centroid_y
    return df

def calculate_angle(df):
    df['normal_magnitude'] = np.sqrt(df['normal_x']**2 + df['normal_y']**2)
    df['velocity_magnitude'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    df['dot_product'] = df['normal_x']*df['velocity_x'] + df['normal_y']*df['velocity_y']
    df['angle'] = np.arccos(df['dot_product'] / (df['normal_magnitude'] * df['velocity_magnitude']))
    return df

def bin_cells(df, bin_size):
    df['bin'] = pd.cut(df['distance_to_front_cell'], bins=np.arange(0, df['distance_to_front_cell'].max() + bin_size, bin_size), include_lowest=True)
    return df

def calculate_average_projections(df):
    df['projection'] = df['dot_product'] / df['normal_magnitude']
    average_projections = df.groupby('bin')['projection'].mean()
    std_proj = df.groupby('bin')['projection'].std()
    return average_projections, std_proj

def plot_velocity_map(df):
    fig, ax = plt.subplots()
    ax.quiver(df['position_x'], df['position_y'], df['velocity_x'], df['velocity_y'], color='black')
    ax.set_title('2D velocity map')
    plt.show()

def plot_radial_velocity(ax, average_projections, std_proj, bin_size, **kwargs):

    rads = [x*bin_size for x in range(0,len(average_projections))]
    ax.errorbar(rads, average_projections, yerr=std_proj, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    ax.set_xlabel('Distance from the centroid')
    ax.set_ylabel('Normal velocity magnitude')
    ax.set_title('Instant radial velocity')
    plt.show()


def run_3d_is_at_front(df):
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