import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from physicell_tools import pyMCDS
import pandas as pd
from physicell_tools.get_perifery import front_cells
from physicell_tools.leastsquares import leastsq_circle

def get_cell_df(pymcds: pyMCDS.pyMCDS):
    """
    Get cell_df from pyMCDS object
    """
    cell_df = pymcds.get_cell_df()
    return cell_df


def calculate_distance_to_front_cell(cells, front):
    for i in range(len(front)):
        current_dist = np.sqrt((cells['position_x']-front[i, 0])**2 + (cells['position_y']-front[i, 1])**2)
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
    xc, yc, R, residu = leastsq_circle(front_cell_positions[:, 0], front_cell_positions[:, 1])
    cell_df['distance_to_front_circle'] = (np.sqrt((cell_df['position_x']-xc)**2 + (cell_df['position_y']-yc)**2)).values - R

    cell_df['distance_to_front_cell'] = calculate_distance_to_front_cell(cell_df, front_cell_positions)
    cell_df['distance_to_center'] = np.sqrt((cell_df['position_x'])**2 + (cell_df['position_y'])**2)
    return


if __name__ == '__main__':

    sims = range(0, 1000, 1)
    distance_to_front_cell =  []
    distance_to_front_circle = []
    min_distance_to_front_cell = []
    for sim in sims:
        # pymc = pyMCDS.pyMCDS('final.xml' ,f'../../data/raven_22_06_patient_sims/PhysiCell_{sim}/output')
        cell_info = pd.read_hdf('./../../data/data_2306_presims.h5', key=f'PhysiCell_{sim}')
        positions, types = front_cells(cell_info)
        type_1_cells = cell_info[cell_info['cell_type'] == 1]

        calculate_distance_to_front(type_1_cells, positions)

        unique_clones = type_1_cells['clone_ID'].unique()
        for clone in unique_clones:
            single_clone = type_1_cells[type_1_cells['clone_ID']==clone]

            # get average distance to front
            mean_dist_to_front_circle = single_clone['distance_to_front_circle'].mean()
            mean_dist_to_front_cell = single_clone['distance_to_front_cell'].mean()
            min_dist_to_front_cell = single_clone['distance_to_front_cell'].min()

            # add to list
            distance_to_front_circle.append(mean_dist_to_front_circle)
            distance_to_front_cell.append(mean_dist_to_front_cell)
            min_distance_to_front_cell.append(min_dist_to_front_cell)

    # plot histogram

    fig1, ax1 = plt.subplots()
    sns.histplot(data=distance_to_front_circle, ax=ax1, bins=20)
    ax1.set_title('Mean distance to front circle')

    fig2, ax2 = plt.subplots()
    sns.histplot(data=distance_to_front_cell, ax=ax2, bins=20)
    ax2.set_xlabel('Distance to front')
    ax2.set_ylabel('Count')
    ax2.set_title('Mean distance to front cell')

    fig3, ax3 = plt.subplots()
    sns.histplot(data=min_distance_to_front_cell, ax=ax3, bins=20)
    ax3.set_title('Min distance to front cell')

    plt.show()