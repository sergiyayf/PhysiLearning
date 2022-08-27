import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from palettable.colorbrewer.sequential import YlOrRd_9 as col
from matplotlib.colors import ListedColormap
import matplotlib 
plt.rcParams.update({'font.size': 12,
                    'font.weight': 'normal'})
#plt.style.use('default')
plt.style.use('dark_background')
cmap = col.mpl_colors; 
def get_files_to_process(time_when_switch=30, cluster='raven'):
    """
    cluster 1  - raven
    cluster 2 - cobra
    
    function to get the list of files, from some of the simulations sets
    """
    if cluster == 'raven': 
        files = []
                
        for k in range(1,11):
            q = time_when_switch;
            
            filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\\20210524_raven\h5_data_folder_raven\\19_05_raven_switch_at_'+str(q)+'_run_'+str(k)+'\\all_data.h5';
            files.append(filename);
        return files;  
    elif cluster == 'cobra': 
        files = [];
        for k in range(1,11):
            q = time_when_switch;
           
            filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\\20210524_cobra\h5_data_cobra_19_05\\19_05_cobra_switch_at_'+str(q)+'_run_'+str(k)+'\\all_data.h5';
            files.append(filename);
    return files; 
    
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

def gether_data(files): 
    R, b_freq = [],[]
    for filename in files: 
        
        read_radii = pd.read_hdf(filename,'data/radius')
        #print(read_radii)
        fronts = pd.read_hdf(filename,'data/front_cell_number')
        #print(fronts)
        read_growth_layer = pd.read_hdf(filename,'data/growth_layer');
        #print(read_growth_layer);
        Rad = np.array(read_radii)
        Growth = np.array(read_growth_layer); 
        num_cells = np.array(fronts); 
        blue_frequency = num_cells[:,2]/np.sum(num_cells,1)
        R.append(Rad.flatten()) 
        b_freq.append(blue_frequency.flatten()) 
        
    return np.array(R), np.array(b_freq) 

def get_means(files):
    R, b = gether_data(files)
    length = []
    for element in R:
        length.append(len(element))

    min_sim = min(length)
    cut_blue,cut_R = [],[]
    for blue, Radius in zip(b,R):
        
        blue = blue[range(min_sim)]
        Radius = Radius[range(min_sim)]
        
        cut_blue.append(blue)
        cut_R.append(Radius)
   
    return np.mean(cut_R,axis=0),np.std(cut_R,axis =0) ,np.mean(cut_blue,axis=0),np.std(cut_blue,axis =0) 

def fit_line(x,y,c): 
    new_x = x[~np.isnan(y)]
    new_y = y[~np.isnan(y)] 
   
    coef = np.polyfit(new_x,new_y,1) 
    poly1d_fn = np.poly1d(coef) 
    # poly1d_fn is now a function which takes in x and returns an estimate for y
    
    slope = 10000*coef[0]
    ax1.plot(new_x, poly1d_fn(new_x), '--w')
    ax1.text(new_x[-1],poly1d_fn(new_x[-1]),'%.1f'% slope, color = 'w')
    ax2.scatter(new_x[0],coef[0],color = c)
    #print(coef[0]/new_x[0])
    #print(new_x[0])
    #print(coef[0])
    #print('next')
    return 0
 

# main
fig,(ax1,ax2)=plt.subplots(1,2)
k=1
for time in [5,10,15,20]:
    files = get_files_to_process(time_when_switch=time, cluster='cobra')
    R,err_R,b,err_b = get_means(files)
    b[b==0]=np.nan
    ax1.plot(R,b,label=str(time),color=cmap[k])
    fit_line(R,b,cmap[k])
    k+=1
for time in [30,35,40,50]:
    files = get_files_to_process(time_when_switch=time, cluster='raven')
    R,err_R,b,err_b = get_means(files)
    b[b==0]=np.nan
    ax1.plot(R,b,label=str(time),color=cmap[k])
    fit_line(R,b,cmap[k])
    k+=1
#plt.fill_between(R,b-err_b,b+err_b,color = 'c', alpha = 0.5)
ax1.set_ylim(0,0.1)
ax1.set_title('Frequency')
ax1.set_ylabel('Blue frequency') 
ax1.set_xlabel('Rad')

plt.legend() 
plt.show()
