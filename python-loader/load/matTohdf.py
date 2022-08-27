import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from load.pyMCDS import pyMCDS
import numpy as np
import pathlib
import os
import matplotlib as mpl
import pathlib
from auxiliary import get_perifery
from auxiliary import get_perifery
import matplotlib.pyplot as plt
from auxiliary import leastsquares as lsq
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from plot import density
from matplotlib import cm
import pandas as pd 

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def load_data(direct):
    # define the path
    directory = pathlib.Path(direct)

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
    for eachfile in files[::]:
        mc = pyMCDS(eachfile, directory)
        mcdss.append(mc)
        

    return mcdss


def matTohdf5(directory_path):
    """
    Input path directory of the output matlab files from the PhysiCell simulations. 
    
    This code reads all the files and restores all the data into one .h5 file wich is much easier to handle, opens quickly and allows to replot all the data very efficiently.
    
    ----------
    data is stored in the same directory as original data. 
    
    h5 file has structure: 
    
    data\cells
                containts 'cell'+str(i) all cell information from the matlab file 
    data\\radius   
                contains radii of the colony for each timepoint 
    data\\front_cell_number
                number of cells at the front, for different 3 types. if there are more cell types code needs to be improved
    data\growth_layer 
                contains growth layer widths for each time point
                
    
    """

    # import the data from the 'output*.xml' files from PhysiCell
    h5File = directory_path+'\\all_data.h5'# "D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\\20210401_rad_tests\\01_04_rad_test_500\output\\all_data.h5";
    data = load_data(directory_path)
     
    # get number of time steps
    num_steps= len(data)

    num_types = 3; 
    cyan_cell_type = 2;
    
    # define some variables
    num_cells = np.zeros([num_steps, num_types], dtype='int') 
    radii = np.zeros(num_steps)
    growth_layers = np.zeros(num_steps)
    history = []
    color_history = []

    for time_index in range(num_steps):
        
        # load cell population, and store a copy
        current_population = data[time_index].get_cell_df()
        copy_population = current_population; 
        copy_population.to_hdf(h5File,'data/cells/cell'+str(time_index));
            
        #print(np.unique(current_population['current_phase']))
        
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
                
        # change to polar coordinates to calculate the growth_layer_width
        rho,theta = cart2pol(x_growth,y_growth);
        # choose some angle part 
        thetas_chosen = theta[theta > 3.0]; 
        rhos_chosen = rho[theta>3.0]; 
                                
        indx = np.argsort(rhos_chosen);
        rho = rhos_chosen[indx]; 
        theta = thetas_chosen[indx]; 
        
        growth_layer = - min(rho) + max(rho); 
        growth_layers[time_index] = growth_layer;
        
        #print('growth layer = '+str(growth_layer));
                
        # get cells on the perifery 
        positions, types = get_perifery.FrontCells(current_population);
        
        # store fronts
        front_pos_df = pd.DataFrame(positions); 
        front_types_df = pd.DataFrame(types); 
        
        front_pos_df.to_hdf(h5File,'data/fronts/positions/time'+str(time_index));
        front_types_df.to_hdf(h5File, 'data/fronts/types/time'+str(time_index)); 
        
        rows = positions[:,0];
        cols = positions[:,1];
        [xc,yc,Rad, res] = lsq.leastsq_circle(rows,cols) 
       
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
        
        print(str(time_index)+' out of '+str(num_steps)+' front calculation ' );
        
        # add tracking of non-yellow: 
        pos_of_not_yellow = positions[types != 0 ]; 
        not_yellow = current_population[current_population['cell_type'] != 0 ]; 
        not_yellow.to_hdf(h5File, 'data/non_yellow/time'+str(time_index));
                          
        ancestors = []; 
        cell_color = []; 
        
        for each_not_yellow_at_perifery in pos_of_not_yellow:
            this_not_yellow = not_yellow[not_yellow['position_x'] == each_not_yellow_at_perifery[0]];
            
            cell_color.append(this_not_yellow['cell_type'].values[0]);
            
            mom_of_this_not_yellow = not_yellow[not_yellow['ID']==not_yellow['parent_ID'].values[0]];
            this_not_yellow2 = this_not_yellow
            while not not_yellow[not_yellow['ID']==this_not_yellow2['parent_ID'].values[0]].empty:
                mom_of_this_not_yellow = not_yellow[not_yellow['ID']==this_not_yellow2['parent_ID'].values[0]];
                this_not_yellow2 = mom_of_this_not_yellow; 
            if mom_of_this_not_yellow.empty: 
                    mom_of_this_not_yellow = this_not_yellow;
                    
            oldest_ancestor = mom_of_this_not_yellow['ID'].values[0]; 
            ancestors.append(oldest_ancestor); 
        
        # add tracking of blue at the front       
        #pos_of_blues = positions[types == cyan_cell_type]; 
        #only_blue = current_population[current_population['cell_type'] == cyan_cell_type];
        #only_blue.to_hdf(h5File,'data/only_blue/cell'+str(time_index));
        
        #ancestors = []; 
        
        #for each_blue_at_perifery in pos_of_blues: 
            #this_blue = only_blue[only_blue['position_x'] == each_blue_at_perifery[0] ]; 
            
            #mom_of_this_blue = only_blue[only_blue['ID']==this_blue['parent_ID'].values[0]];
            #this_blue2 = this_blue
            #while not only_blue[only_blue['ID']==this_blue2['parent_ID'].values[0]].empty:
                #mom_of_this_blue = only_blue[only_blue['ID']==this_blue2['parent_ID'].values[0]];
                #this_blue2 = mom_of_this_blue; 
                
            
            #if mom_of_this_blue.empty: 
                    #mom_of_this_blue = this_blue;
                    
            #oldest_ancestor = mom_of_this_blue['ID'].values[0]; 
            #ancestors.append(oldest_ancestor); 
        
        history.append(np.array(ancestors));
        color_history.append(np.array(cell_color));
    #print('num_cells');
    #print(num_cells); 
    cell_number_df = pd.DataFrame(num_cells, columns = ['type0', 'type1', 'type2']);
    cell_number_df.to_hdf(h5File,'data/front_cell_number');
    
    rad_df = pd.DataFrame(radii, columns = ['Rad']);
    growth_layer_df = pd.DataFrame(growth_layers, columns = ['growth_layer_width']);

    rad_df.to_hdf(h5File,'data/radius');
    growth_layer_df.to_hdf(h5File,'data/growth_layer');
    
    history_df = pd.DataFrame(history)
    history_df.to_hdf(h5File,'data/history');
    
    color_history_df = pd.DataFrame(color_history)
    color_history_df.to_hdf(h5File,'data/color_history');
    
   
    

#directory_path = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\\20210401_rad_tests\\01_04_rad_test_500\output'
#matTohdf5(directory_path)
