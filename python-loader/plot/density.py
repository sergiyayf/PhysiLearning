from load.pyMCDS import pyMCDS
import numpy as np
import matplotlib.pyplot as plt
import glob 
import pathlib
from matplotlib import cm
from matplotlib.colors import Normalize 
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def using_mpl_scatter_density(fig, x, y, color_map):
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=color_map)
    fig.colorbar(density, label='Number of points per pixel')
    
    
def get_single_density_plot(cells, colormap, xlim = [-5000,5000], ylim = [-5000,5000]):
    """ This function gets cells dataframe and colormap as an imput and scatter plots cells """ 
    fig = plt.figure()
    using_mpl_scatter_density(fig, cells['position_x'] , cells['position_y'], colormap)
    plt.xlim(xlim);
    plt.ylim(ylim);


def get_type_dependent_density_plot(cells_list, colormap_list, xlim = [-5000,5000], ylim = [-5000,5000]):
    fig = plt.figure()
    for cell, cmap  in zip(cells_list,colormap_list): 
        using_mpl_scatter_density(fig, cell['position_x'] , cell['position_y'], cmap)
    plt.xlim(xlim);
    plt.ylim(ylim);
