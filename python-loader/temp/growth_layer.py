from load.pyMCDS import pyMCDS
import numpy as np
import pathlib
import os
import matplotlib as mpl
import pathlib
from auxiliary import get_perifery
import matplotlib.pyplot as plt
from auxiliary import leastsquares as lsq
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from plot import density
from matplotlib import cm
import pandas as pd 

class visualize:
    """contains all the plotting functions"""

    def __init__(self, x, rows, cols):
        self.rows = rows
        self.cols = cols
        self.x = x

        #plt.close('all')
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
    directory = pathlib.Path('D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\\20210401_rad_tests\\01_04_rad_test_500\output')

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
    for eachfile in files[0:12:1]:
        mc = pyMCDS(eachfile, directory)
        mcdss.append(mc)
        

    return mcdss

# set some manual things
#types = ['rep:1 adh:1','rep:0.5 adh:2','rep:2 adh:0.5']
types = ['wt','rd', 'rn']
color_order = 'yrc'
#num_types = len(types)  # number of cell types in the sim (stored in parameter 'cell_type')

# import the data from the 'output*.xml' files from PhysiCell
h5File = "all_data.h5";
data = load_data()
      

#df0 = data[0].get_cell_df()
#df1 = data[1].get_cell_df()
#print(df0)
#df0.to_hdf(h5File, "/data/d0");
#df1.to_hdf(h5File, "/data/d1");

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


# get number of time steps
num_steps=len(data)

num_types = 3; 
# loop through all time points to extract values of interest (such as number of cell per type etc.)
num_cells = np.zeros([num_steps, num_types], dtype='int')  #this is where the nomber of
radii = np.zeros(num_steps)
growth_layers = np.zeros(num_steps)


for time_index in range(num_steps):
    
    # load cell population, and store a copy
    current_population = data[time_index].get_cell_df()
    copy_population = current_population; 
    copy_population.to_hdf(h5File,'data/cells/cell'+str(time_index));
        
    print(np.unique(current_population['current_phase']))
    
    # separate phases
    G0G1_phase_cells = current_population[current_population['current_phase']==4];
    G2_phase_cells = current_population[current_population['current_phase']==10];
    S_phase_cells = current_population[current_population['current_phase']==12];
    M_phase_cells = current_population[current_population['current_phase']==13];
    non_G0_phase_cells = current_population[current_population['current_phase']!=4];
    
    # make sure there are non G0G1 cells, because there could be none initially 
    if len(non_G0_phase_cells)>2:
        x_growth = np.array(non_G0_phase_cells['position_x']);
        y_growth = np.array(non_G0_phase_cells['position_y']); 
    else:
        x_growth = np.array(current_population['position_x']);
        y_growth = np.array(current_population['position_y']); 
    print(len(x_growth))
    
    # change to polar coordinates to calculate the growth_layer_width
    rho,theta = cart2pol(x_growth,y_growth);
    # choose some angle part 
    thetas_chosen = theta[theta > 3.0]; 
    rhos_chosen = rho[theta>3.0]; 
    print(len(rhos_chosen))
    
    indx = np.argsort(rhos_chosen);
    rho = rhos_chosen[indx]; 
    theta = thetas_chosen[indx]; 
     
    growth_layer = - min(rho) + max(rho); 
    growth_layers[time_index] = growth_layer;
    print('growth layer = '+str(growth_layer));
       
    #x,y = pol2cart(rho,theta);
    #cells_list = [G0G1_phase_cells,non_G0_phase_cells];
    #colormap_list = [new_red_cmp, new_yellow_cmp]; 
    #plt.figure()
    #density.get_type_dependent_density_plot(cells_list, colormap_list)
    #plt.scatter(x,y);
    
    # get cells on the perifery 
    positions, types = get_perifery.FrontCells(current_population);
    
    ##plt.figure()
    rows = positions[:,0];
    cols = positions[:,1];
    [xc,yc,Rad, res] = lsq.leastsq_circle(rows,cols) 
    ##lsq.plot_data_circle(rows, cols, xc, yc, Rad)
    #Rad = max(rho);
    radii[time_index] = Rad; 
    
    values, counts = np.unique(types, return_counts = True); 
   
    type0_counts = counts[values==0.]; 
    #print(type0_counts); 
    #print(len(type0_counts));
    if len(type0_counts)==0:
        type0_counts = [0.];
   
    type1_counts = counts[values==1.]; 
    #print(type1_counts); 
    #print(len(type1_counts));
    if len(type1_counts)==0:
        type1_counts = [0.];
        
    type2_counts = counts[values==2.]; 
    #print(type2_counts); 
    #print(len(type2_counts));
    if len(type2_counts)==0:
        type2_counts = [0.];
  
    counts = [type0_counts[0],type1_counts[0],type2_counts[0]];
    #print(counts)
    ##count, division=np.histogram(current_population['cell_type'],num_types)
    num_cells[time_index, :] = counts
    
    print(str(time_index)+'out of'+str(num_steps));
    
print('num_cells');
print(num_cells); 
cell_number_df = pd.DataFrame(num_cells, columns = ['type0', 'type1', 'type2']);
cell_number_df.to_hdf(h5File,'data/front_cell_number');
## plot results
#steps = range(num_steps) # used as x-coordinate

#SummaryPlot = visualize(radii/radii[0],1,2) # args: x coordinate, num of rows, num of cols

#plt.close('all');
#SummaryPlot.numbers(1, 'linear', 'linear')
#SummaryPlot.frequencies(2, 'linear', 'linear')

#SummaryPlot = visualize(radii,1,2) # args: x coordinate, num of rows, num of cols
#SummaryPlot.numbers(1, 'linear', 'linear')
#SummaryPlot.frequencies(2, 'linear', 'linear')

rad_df = pd.DataFrame(radii, columns = ['Rad']);
growth_layer_df = pd.DataFrame(growth_layers, columns = ['growth_layer_width']);

rad_df.to_hdf(h5File,'data/radius');
growth_layer_df.to_hdf(h5File,'data/growth_layer');
#print(rad_df) 

plt.figure()
plt.plot(radii,growth_layers);

plt.show()
