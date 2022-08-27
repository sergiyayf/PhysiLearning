from load.pyMCDS import pyMCDS
import numpy as np
import matplotlib.pyplot as plt
import glob 
import pathlib
from plot import density
from matplotlib import cm
from matplotlib.colors import Normalize 
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# define the path

directory = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\26_10_mechanoevolution_2\output'
#directory = pathlib.Path('D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\26_10_mechanoevolution_2\output')
# choose a file to process
mcds1 = pyMCDS('output00000070.xml', directory)


last_cell = mcds1.get_cell_df()
density.get_single_density_plot(last_cell, plt.cm.Reds) 

#define new colormaps with transparent 0 value, this is needed for type dependent density plot not to overplot other point
reds = cm.get_cmap('Reds',256); 
new_reds = reds(np.linspace(0,1,256))
empty = np.array([0,0,0,0])
new_reds[0,:] = empty;
new_red_cmp = ListedColormap(new_reds)

yellows = cm.get_cmap('YlOrBr',256); 
new_yellows = yellows(np.linspace(0,1,256))
empty = np.array([0,0,0,0])
new_yellows[0,:] = empty;
new_yellow_cmp = ListedColormap(new_yellows)

#plot all cells not differentiating by the type 
#fig = plt.figure()
#using_mpl_scatter_density(fig, last_cell['position_x'] , last_cell['position_y'], plt.cm.Reds)

type0_cells = last_cell[last_cell['cell_type'] == 0]; 
type1_cells = last_cell[last_cell['cell_type'] == 1];
type2_cells = last_cell[last_cell['cell_type'] == 2];


cells_list = [type0_cells, type1_cells]; 
colormap_list = [new_red_cmp, new_yellow_cmp]; 

density.get_type_dependent_density_plot(cells_list, colormap_list)

plt.show()
