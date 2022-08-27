import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

import matplotlib 
from matplotlib.markers import MarkerStyle
from matplotlib.collections import EllipseCollection

custom_blue =(82/255,175/255,230/255)
custom_red =(190/255,28/255,45/255)

def plot_cells(ax = None, c_type = 0, color = 'grey',edgecolor='dimgrey'):
    circles = [plt.Circle((xi,yi), radius = ri, linewidth = 0) for xi,yi,ri in zip(x_pos[cell_type==c_type],y_pos[cell_type==c_type],rad[cell_type==c_type])]
    c = matplotlib.collections.PatchCollection(circles,color=color, alpha=0.75, edgecolor = edgecolor,linewidth=0.01)
    ax.scatter(x_pos[cell_type==c_type],y_pos[cell_type==c_type],color = color, s=1.5, alpha = 0.75, linewidth = 0.00, edgecolor = edgecolor )
    
    ax.add_collection(c)
    ax.set_xlim(-6000,6000)
    ax.set_ylim(-6000,6000)
    ax.set_aspect('equal')
    return

def plot_colony():
    plot_cells(ax = ax, color =[0.25,0.25,0.25,0.5])
    plot_cells(ax=ax, c_type = 1, color = custom_red, edgecolor = 'red')
    plot_cells(ax=ax, c_type = 2, color = custom_blue, edgecolor = 'deepskyblue')
    return

# Read file
i = 1
filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\2021_12_13_random_switch_part_1\13_12_2021_random_switching_run_2\output\all_data.h5'
#filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20210524_raven\19_05_raven_switch_at_30_run_1\output\all_data.h5'
TitleName = 'run3';

# get positions and cell type 
f = h5py.File(filename, 'r')
dat = f['data/cells'];
number_of_frames = len(dat.keys())


timeframe = 98;
for timeframe in range(20,101):

    cells = pd.read_hdf(filename,'data/cells/cell'+str(timeframe));
    x_pos = cells['position_x']
    y_pos = cells['position_y']
    cell_type = cells['cell_type']
    rad = (cells['total_volume']*3/4/np.pi)**(1/3)


    fig, ax = plt.subplots(figsize=(10,10),dpi=300)

    plot_colony()

    plt.savefig(r'images\random_switch\colony'+str(timeframe)+'.png')
    plt.clf()
    plt.close()

plt.show()
