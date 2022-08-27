from load.pyMCDS import pyMCDS
import numpy as np
import matplotlib.pyplot as plt
import glob 
import pathlib
from auxiliary import get_perifery

# define the path
directory = pathlib.Path(r'C:\Users\saif\Desktop\results_sims_temp\27_01_small_colony_timing_test_1\output')

# choose a file to process
mcds1 = pyMCDS('output00000005.xml', directory)

# get real cells dataframe and positions to analyse
clls = mcds1.get_cell_df();

get_perifery.VisualizeFront(clls)

positions, types = get_perifery.FrontCells(clls);

print(positions)
print(types)
plt.show()
