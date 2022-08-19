from python_loader.pyMCDS import pyMCDS
import numpy as np

def get_PC_output(file='output00000111.xml',dir=r'C:\Users\saif\Desktop\Serhii\Projects\hackathon_PhysiCell\PhysiCell_V_1.10.4\output_1'
                  ,resistant = 1, susceptible = 0):
    """

    Parameters
    ----------
    file - string, filename to of simulation to get output from xml
    dir - string, directory of the simulation

    Returns
    -------
    dictionary  resistant or susceptible cell numbers

    """

    mcds = pyMCDS(file,dir)
    cell_df = mcds.get_cell_df()

    cell_types = np.array(cell_df['cell_type'])
    un_types, un_number = np.unique(cell_types,return_counts=True)
    output_dict = {'susceptible': un_number[un_types==susceptible],
                   'resistant': un_number[un_types==resistant]}
    return output_dict

if __name__ == '__main__':
    out = get_PC_output()
    print(out['susceptible'])