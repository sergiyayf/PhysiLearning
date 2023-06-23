import numpy as np

from physicell_tools import pyMCDS
import pandas as pd
from physicell_tools.get_perifery import front_cells


def get_cell_df(pymcds: pyMCDS.pyMCDS):
    """
    Get cell_df from pyMCDS object
    """
    cell_df = pymcds.get_cell_df()
    return cell_df

if __name__ == '__main__':
    sims = range(0, 10, 1)
    h5file = 'data.h5'

    for sim in sims:
        pymc = pyMCDS.pyMCDS('final.xml' ,f'./../data/raven_22_06_patient_sims/PhysiCell_{sim}/output')
        cell_info = get_cell_df(pymc)

        positions, types = front_cells(cell_info)
        current_population = cell_info
        cells_at_front = pd.DataFrame()
        current_population['is_at_front'] = np.zeros_like(current_population['position_x'])
        # if position is in front, set is_at_front to 1
        current_population.loc[current_population['position_x'].isin(positions[:, 0]), 'is_at_front'] = 1
        simplified = current_population[['ID', 'parent_ID', 'clone_ID',
                                         'position_x', 'position_y', 'position_z',
                                         'is_at_front', 'cell_type', 'elapsed_time_in_phase',
                                         'total_volume', 'pressure']]


        #save to h5
        #cell_info.to_hdf(h5file, key=f'PhysiCell_{sim}')