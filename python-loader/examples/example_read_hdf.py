import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

import matplotlib 
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

class visualize:
    """contains all the plotting functions"""

    def __init__(self, x, rows, cols):
        self.rows = rows
        self.cols = cols
        self.x = x

        #plt.close('all')
        plt.figure(figsize=(8, 8))
        
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
        self.create_plot(y, plotIndex, xscale, yscale, xlabel='R', ylabel='number of cells')

    def frequencies(self, plotIndex, xscale='linear', yscale='linear'):
        y = num_cells/np.sum(num_cells, 1)[:, None]
        self.create_plot(y, plotIndex, xscale, yscale, xlabel='R', ylabel='frequencies')
    
    def GrowthLayerPlot(self, plotIndex, xscale='linear', yscale='linear'):
        y = Growth; 
        self.create_plot(y, plotIndex, xscale, yscale, xlabel='R', ylabel='growth layer')


###

filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\20210524_raven\h5_data_folder_raven\19_05_raven_switch_at_30_run_3\all_data.h5'; 
TitleName = 'run3';
f = h5py.File(filename, 'r')

#print(list(f['data'].keys()))

dat = f['data/cells'];
cells_look_info = pd.read_hdf(filename,'data/cells/cell1');
print(cells_look_info)


read_radii = pd.read_hdf(filename,'data/radius')
#print(read_radii)
fronts = pd.read_hdf(filename,'data/front_cell_number')
#print(fronts)
read_growth_layer = pd.read_hdf(filename,'data/growth_layer');
#print(read_growth_layer);
Rad = np.array(read_radii)
Growth = np.array(read_growth_layer); 
num_cells = np.array(fronts); 

types = ['wt','rd', 'rn']
color_order = 'yrc'

SummaryPlot = visualize(Rad,1,1) # args: x coordinate, num of rows, num of cols

#plt.close('all');
SummaryPlot.numbers(1, 'linear', 'linear')

SummaryPlot = visualize(Rad,1,1)
SummaryPlot.frequencies(1, 'linear', 'linear')

SummaryPlot = visualize(Rad,1,1)
SummaryPlot.GrowthLayerPlot(1);
plt.title(TitleName)

plt.figure()
plt.plot(range(len(Rad)),Rad)
plt.xlabel('time a.u.') 
plt.ylabel('Rad')

R = Rad.flatten()
plt.figure()
plt.plot(range(len(R)-1),np.diff(R))

plt.show()
