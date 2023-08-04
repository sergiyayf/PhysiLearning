import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from physicell_tools import pyMCDS
import pandas as pd
from physicell_tools.get_perifery import front_cells
from physicell_tools.leastsquares import leastsq_circle
from scipy.stats import gaussian_kde


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
    print(R)
    cell_df = cell_df.copy()

    cell_df['distance_to_front_circle'] = (np.sqrt((cell_df['position_x']-xc)**2 + (cell_df['position_y']-yc)**2)).values - R
    cell_df['distance_to_front_cell'] = calculate_distance_to_front_cell(cell_df, front_cell_positions)
    cell_df['distance_to_center'] = np.sqrt((cell_df['position_x'])**2 + (cell_df['position_y'])**2)

    return cell_df


if __name__ == '__main__':

    sims = range(0, 1000, 1)
    #sims = [80]
    mutation_rates = [0.000825, 0.00086, 0.00087, 0.00088, 0.0009, 0.001, 0.000875]
    # file_list = ['data/simplified_data_1107_mutation_rate_0825.h5',
    #                 'data/simplified_data_1207_mutation_rate_086.h5',
    #                 'data/simplified_data_1207_mutation_rate_087.h5',
    #                 'data/simplified_data_1207_mutation_rate_088.h5',
    #                 'data/simplified_data_1107_mutation_rate_09.h5',
    #                 'data/simplified_data_0707_presims.h5',
    #                 'data/simplified_data_1207_mutation_rate_0875.h5']
    file_list = ['data/simplified_data_1107_mid_mutation_rate.h5']
    #
    for file in file_list:
        distance_to_front_cell = []
        distance_to_front_circle = []
        min_distance_to_front_cell = []
        distance_to_center = []
        for sim in sims:
            # pymc = pyMCDS.pyMCDS('final.xml' ,f'../../data/raven_22_06_patient_sims/PhysiCell_{sim}/output')
            cell_info = pd.read_hdf(file, key=f'PhysiCell_{sim}')
            type_1_cells = cell_info[cell_info['cell_type'] == 1]

            cells_at_front = cell_info[cell_info['is_at_front'] == 1]
            positions = cells_at_front[['position_x', 'position_y']].values

            type_1_cells = calculate_distance_to_front(type_1_cells, positions)

            unique_clones = type_1_cells['clone_ID'].unique()
            for clone in unique_clones:
                single_clone = type_1_cells[type_1_cells['clone_ID']==clone]

                # get average distance to front
                mean_dist_to_front_circle = single_clone['distance_to_front_circle'].mean()
                mean_dist_to_front_cell = single_clone['distance_to_front_cell'].mean()
                min_dist_to_front_cell = single_clone['distance_to_front_cell'].min()
                distance_to_center_max = single_clone['distance_to_center'].max()

                # add to list
                distance_to_front_circle.append(mean_dist_to_front_circle)
                distance_to_front_cell.append(mean_dist_to_front_cell)
                min_distance_to_front_cell.append(min_dist_to_front_cell)
                distance_to_center.append(distance_to_center_max)

        # save distributions as dataframe to hdf5 file
        # df = pd.DataFrame({ 'min_distance_to_front_cell': min_distance_to_front_cell,
        #                     'distance_to_center': distance_to_center})
        # df.to_hdf('./data/distributions.h5', key=file)

    # print statistics, mean, median and 25 and 75 percentiles
    print(f'Mean distance to front cell (min): {np.mean(min_distance_to_front_cell)}')
    print(f'Median distance to front cell (min): {np.median(min_distance_to_front_cell)}')
    # print(f'25 percentile distance to front cell (min): {np.percentile(min_distance_to_front_cell, 25)}')
    # print(f'75 percentile distance to front cell (min): {np.percentile(min_distance_to_front_cell, 75)}')
    # print(f'95 percentile distance to front cell (min): {np.percentile(min_distance_to_front_cell, 95)}')
    #
    #
    # fig3, ax3 = plt.subplots()
    # sns.histplot(data=min_distance_to_front_cell, ax=ax3, bins=20)
    # ax3.set_title('Distance from the outermost cell of a clone to the closest front cell')
    # ax3.set_xlabel('Distance to front cell')
    # ax3.set_ylabel('Number of clones')
    # #fig3.savefig('./../../data/figures/fig_s1c.png')
    #
    # fig4, ax4 = plt.subplots()
    # sns.histplot(data=distance_to_center, ax=ax4, bins=20)
    # ax4.set_title('Distance from the outermost cell of a clone to the center of the tumor')
    # ax4.set_xlabel('Distance to center')
    # ax4.set_ylabel('Number of clones')
    #
    # # normalize to the radius
    # # get numpy histogram data
    # hist, bin_edges = np.histogram(distance_to_center, bins=20)
    # # normalize to the radius
    # centers = (bin_edges[1:] + bin_edges[:-1])/2
    # hist = hist / (np.pi * centers * 2)
    # # plot
    # fig5, ax5 = plt.subplots()
    # ax5.bar(centers, hist, width=10)
    # ax5.set_title('Normalized')

    # read the distributions from the hdf5 file and plot them in 3d plot
    # with mutation rate on z axis and distribution of min distance over x y
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111, projection='3d')
    # for i, file in enumerate(file_list):
    #     df = pd.read_hdf('./data/distributions.h5', key=file)
    #     hist, bin_edges = np.histogram(df['distance_to_center'], bins=20)
    #     centers = (bin_edges[1:] + bin_edges[:-1])/2
    #     hist = hist / (np.pi * centers * 2)
    #     ax.bar(centers, hist, zs=mutation_rates[i], zdir='y', alpha=0.8, width=10)
    #     ax.set_xlabel('Distance to center')
    #     ax.set_ylabel('Mutation rate')
    #     ax.set_zlabel('Normalized number of clones')
    #
    #     # plot the same with smoothed data for better visualization
    #     # smooth the data with gaussian kde
    #     kde = gaussian_kde(df['distance_to_center'])
    #     x = np.linspace(0, 1000, 100)
    #     y = kde(x)
    #     ax2.plot(x, y, zs=mutation_rates[i], zdir='y', alpha=0.8)
    #
    #

    plt.show()

