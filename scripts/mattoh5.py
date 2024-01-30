# Script to read PhysiCell output and save to h5 file ignoring data I don't need
# and calculating convex hull to get the cells at the front, plus calculating and saving
# some parameters such as radius of the front, number of clones, etc.
import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
from physicell_tools import pyMCDS
from physicell_tools.get_perifery import front_cells
import pandas as pd


def get_cell_df(pymcds: pyMCDS.pyMCDS):
    """
    Get cell_df from pyMCDS object
    """
    cell_df = pymcds.get_cell_df()
    return cell_df


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


if __name__ == '__main__':

    sims = [49]
    distance_to_front_cell = []
    distance_to_front_circle = []
    min_distance_to_front_cell = []
    per_sim_min_distance = []

    for sim in sims:

        pymcds = pyMCDS.pyMCDS('output00000120.xml' ,f'../data/all_presims/sim_{sim}/')
        cell_df = pymcds.get_cell_df()
        cell_df['is_at_front'] = np.zeros_like(cell_df['position_x'])
        cell_df['core_shell'] = np.zeros_like(cell_df['position_x'])
        cell_diameter = 14
        growth_threshold = 1e-5
        # slice through z axis to find cells at front
        z = -600
        while not z > 600:
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
        y = -600
        while not y > 600:
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

        z = -600
        while not z > 600:
            cells_in_slice = non_growing_cells[(non_growing_cells['position_z'] > z) & (non_growing_cells['position_z'] < z + cell_diameter)]
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
        y = -600
        while not y > 600:
            cells_in_slice = non_growing_cells[(non_growing_cells['position_y'] > y) & (non_growing_cells['position_y'] < y + cell_diameter)]
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


        simplified = cell_df[['ID', 'parent_ID', 'clone_ID',
                                         'position_x', 'position_y', 'position_z',
                                         'is_at_front', 'cell_type', 'elapsed_time_in_phase',
                                         'total_volume', 'pressure', 'transition_rate', 'core_shell']]

        simplified.to_hdf('new_presims_3d.h5', key=f'data/cells/sim_{sim}')

        cell_info = simplified
        type_1_cells = cell_info[cell_info['cell_type'] == 1]

        cells_at_front = cell_info[cell_info['is_at_front'] == 1]
        positions = cells_at_front[['position_x', 'position_y', 'position_z']].values

        type_1_cells = calculate_distance_to_front(type_1_cells, positions)

        unique_clones = type_1_cells['clone_ID'].unique()
        _min_per_sim = []
        for clone in unique_clones:
            single_clone = type_1_cells[type_1_cells['clone_ID'] == clone]

            # get average distance to front
            mean_dist_to_front_circle = single_clone['distance_to_front_circle'].mean()
            mean_dist_to_front_cell = single_clone['distance_to_front_cell'].mean()
            min_dist_to_front_cell = single_clone['distance_to_front_cell'].min()

            # save to hdf5 file
            single_clone.to_hdf('new_presims_3d.h5', key=f'data/clones/sim_{sim}/clone_{int(clone)-1}')


            _min_per_sim.append(min_dist_to_front_cell)

            # add to list
            distance_to_front_circle.append(mean_dist_to_front_circle)
            distance_to_front_cell.append(mean_dist_to_front_cell)
            min_distance_to_front_cell.append(min_dist_to_front_cell)

        if len(_min_per_sim) > 0:
            per_sim_min_distance.append(np.min(_min_per_sim))
        else:
            per_sim_min_distance.append(np.nan)

        # calculate average radius of cells at the front
        R = np.mean(np.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2))
        R_std = np.std(np.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2))
        # core shell radius
        core_shell = cell_info[cell_info['core_shell'] == 1]
        core_shell_positions = core_shell[['position_x', 'position_y', 'position_z']].values
        R_core_shell = np.mean(np.sqrt(core_shell_positions[:, 0]**2 + core_shell_positions[:, 1]**2 + core_shell_positions[:, 2]**2))
        R_core_shell_std = np.std(np.sqrt(core_shell_positions[:, 0]**2 + core_shell_positions[:, 1]**2 + core_shell_positions[:, 2]**2))

        # number of clones
        n_clones = len(unique_clones)
        # min clone distance to front
        if len(_min_per_sim) > 0:
            min_clone = np.min(_min_per_sim)
        else:
            min_clone = np.nan

        # put in dataframe
        clone_df = pd.DataFrame({'sim': sim, 'R': R, 'R_std': R_std, 'R_core_shell': R_core_shell, 'R_core_shell_std': R_core_shell_std,
                                 'n_clones': n_clones, 'min_clone_to_front_distance': min_clone}, index=[0])

        clone_df.to_hdf('new_presims_3d.h5', key=f'data/clone_info_sim_{sim}')
