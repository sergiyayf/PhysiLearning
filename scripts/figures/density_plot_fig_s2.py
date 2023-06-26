import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from physicell_tools import pyMCDS
import pandas as pd
from physicell_tools.get_perifery import front_cells
from physicell_tools.leastsquares import leastsq_circle
from matplotlib import cm
from matplotlib.colors import Normalize
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from cycler import cycler


def get_cell_df(pymcds: pyMCDS.pyMCDS):
    """
    Get cell_df from pyMCDS object
    """
    cell_df = pymcds.get_cell_df()
    return cell_df


if __name__ == '__main__':

    sims = range(0, 1000, 1)

    # initialize pandas dataframe
    df = pd.DataFrame()

    fig, ax = plt.subplots(figsize=(10, 10))
    dark2_colors = plt.cm.Dark2.colors
    set3_colors = plt.cm.Set3.colors
    tab20_colors = plt.cm.tab20.colors
    tab20b_colors = plt.cm.tab20b.colors

    # Combine the color palettes into a single cycler
    combined_colors = set3_colors + tab20_colors + tab20b_colors + dark2_colors
    color_cycler = cycler('color', combined_colors)
    ax.set_prop_cycle(color_cycler)
    counter = 0
    for sim in sims:
        # pymc = pyMCDS.pyMCDS('final.xml' ,f'../../data/raven_22_06_patient_sims/PhysiCell_{sim}/output')
        cell_info = pd.read_hdf('./../../data/simplified_data_2306_presims.h5', key=f'PhysiCell_{sim}')
        type_1_cells = cell_info[cell_info['cell_type'] == 1]

        cells_at_front = cell_info[cell_info['is_at_front'] == 1]
        positions = cells_at_front[['position_x', 'position_y']].values

        # concatenate to dataframe
        df = pd.concat([df, type_1_cells])
        unique_clones = type_1_cells['clone_ID'].unique()

        for clone in unique_clones:
            single_clone = type_1_cells[type_1_cells['clone_ID'] == clone]
            ax.scatter(single_clone['position_x'], single_clone['position_y'], s=10, alpha=0.5)
            counter += 1

    ax.set_xlim(-750, 750)
    ax.set_ylim(-750, 750)
    ax.set_title('Density plot of clones at front')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.savefig('./../../data/figures/fig_s2_density.png')

    print("Number of clones: ", counter)
    plt.show()