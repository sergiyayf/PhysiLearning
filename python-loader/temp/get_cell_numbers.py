from load.pyMCDS import pyMCDS
import numpy as np
import pathlib
import os
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askdirectory


class visualize:
    """contains all the plotting functions"""

    def __init__(self, x, rows, cols):
        self.rows = rows
        self.cols = cols
        self.x = x

        plt.close('all')
        plt.figure(figsize=(16, 6))

    def create_plot(self, y, plotIndex, xscale, yscale, xlabel='x', ylabel='y'):
        plt.subplot(self.rows, self.cols, plotIndex)
        for line_index in range(y.shape[1]):
            plt.plot(self.x, y[:, line_index], color_order[line_index], linewidth=2)
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend(types)

    def numbers(self, plotIndex, xscale='linear', yscale='linear'):
        y = num_cells
        self.create_plot(y, plotIndex, xscale, yscale, xlabel='steps', ylabel='number of cells')

    def frequencies(self, plotIndex, xscale='linear', yscale='linear'):
        y = num_cells/np.sum(num_cells, 1)[:, None]
        self.create_plot(y, plotIndex, xscale, yscale, xlabel='steps', ylabel='frequencies')


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def load_data():
    # define the path
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    dir_name = askdirectory(initialdir='/Users/jkayser/MPL_local/Data/Physicell_Serhii/interecation_strength_test/test_1_1003') # show an "Open" dialog box and return the selected path
    directory = pathlib.Path(dir_name)

    # define the pattern
    pattern = "output*.xml"

    # list files with pattern in the directory
    files = []
    for output in directory.glob(pattern):

        a = os.path.basename(output)
        files.append(a)

    # make sure file list is sorted
    files.sort()

    # load data
    mcdss = []
    for eachfile in files[::10]:
        mc = pyMCDS(eachfile, directory)
        mcdss.append(mc)

    return mcdss

# set some manual things
#types = ['rep:1 adh:1','rep:0.5 adh:2','rep:2 adh:0.5']
types = ['rep:0.5 adh:const','rep:2 adh:const']
color_order = 'yrk'
num_types = len(types)  # number of cell types in the sim (stored in parameter 'cell_type')

# import the data from the 'output*.xml' files from PhysiCell
data = load_data()

# get number of time steps
num_steps=len(data)

# loop through all time points to extract values of interest (such as number of cell per type etc.)
num_cells = np.zeros([num_steps, num_types], dtype='int')  #this is where the nomber of
for time_index in range(num_steps):
    current_population = data[time_index].get_cell_df()
    count, division=np.histogram(current_population['cell_type'],num_types)
    num_cells[time_index, :] = count

# plot results
steps = range(num_steps) # used as x-coordinate
SummaryPlot = visualize(steps,1,2) # args: x coordinate, num of rows, num of cols
SummaryPlot.numbers(1, 'linear', 'linear')
SummaryPlot.frequencies(2, 'linear', 'linear')
plt.show()
