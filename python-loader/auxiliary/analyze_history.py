import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from auxiliary import get_perifery


def mask(ID_history, cell_type): 
    """
    mask the width with cell type 
    
    Parameters:
    ----------
    
    ID_history - pandas dataframe containing IDs, loaded from h5 file dataset from the group "data/history"
    
    cell_type - pandas dataframe with clone type history, loaded from h5 file dataset from the group "data/color_history"
    
    
    Returns:
    -------
    
    list of masked histories. 
    
    """
    
    # fill nans of clone IDs with zero to have only floats 
    ID_history = ID_history.fillna(value=0.)
    
    # get the number of unique cell types in the simulation
    types = np.unique(cell_type); 
    types = types[~np.isnan(types)]
    
    masked_history = []
    # loop through different types of cells and create masks that are then multiplied with histories 
    for idx, clone in enumerate(types): 
        temp = cell_type.copy()
        temp[temp!=clone] = 0.
        temp[temp==clone] = 1.
        
        masked_history.append(np.array(ID_history * temp.values ))
        
     
    return masked_history
    

def measure_width(masked_history):
    """
    Calculate clone width in cell numbers
    
    Parameters:
    ----------
    
    masked_history - list of arrays of clone histories
    
    Returns:
    -------
    
    List of dataframes with width instead of clone ID 
    
    """
    
    # initialize list for storing
    width = []
    
    for idx, subclones in enumerate(masked_history): 
        HistDf = pd.DataFrame() 
       
        for time in range(len(subclones)): 
            # get number of unique suclones and number of cells
            (unique_clone, cell_number) = np.unique(subclones[time], return_counts = True) 
            # get a string of unique IDs to make dictionary to save it to dataframe
            string_of_unique_clone_ids = [str(ID) for ID in unique_clone]
            dictionary_of_histories = {ID : [width] for ID, width in zip(string_of_unique_clone_ids, cell_number) }
            one_frame_hist = pd.DataFrame.from_dict(dictionary_of_histories)
            # concatenate different timepoints
            HistDf=pd.concat([HistDf,one_frame_hist],axis = 0, ignore_index = True) 
        # append to the list different cell type width histories   
        
        HistDf = HistDf.drop(columns=['0.0'],axis=1)
        width.append(HistDf)
        
    return width 

def sort(width): 
    """ 
    Sort width histories with respect to the lifetime
    
    Parameters: 
    ----------
    
    width - list of pandas dataframes with clonal width, columns - clone ID, raws - timepoints
    
    Returns: 
    -------
    
    sorted dataframes
    
    """
    lensdf = []
    
    # add two dataframes if their length is larger than 1
    if len(width) !=1:
        total = [width[0].add(width[1], fill_value=0)]
        
    for idx, df in enumerate(total): 
        lens = []
                
        for key in df.keys():
            # calculate length of a clone
            max_not_nan = max(np.where(~np.isnan(df[key].values))[0])
           
            lens.append(max_not_nan)
            #lens.append(np.sum(~np.isnan(df[key].values)))
        # put length into dictionary     
        dictionary = {str(k): [length] for k,length in zip(df.keys(),lens) }
        
        lensdf.append(pd.DataFrame.from_dict(dictionary)) 
    
    if len(lensdf)!=1:
        for i in range(len(lensdf)-1):
            
            # add colorful clones up if there are some 
            combined_df = lensdf[i].add(lensdf[i+1], fill_value = 0)
                        
        # sort for length    
    else:
        combined_df = lensdf[0]
   
    combined_df = combined_df.sort_values(by=0,axis = 1) 
           
    return combined_df.keys() 
    
   
def plot_sorted(ax,sorted_ids, dfs): 
    """
    plot sorted histories
    
    Parameters: 
    ----------
    
    sorted_ids - dataframe containing sorted ids as keys
    """
    #loop through sorted indecies   
    custom_blue =(82/255,175/255,230/255)
    custom_red =(190/255,28/255,45/255) 
    color = [custom_red,custom_blue,'y','k']
    
    labels = np.linspace(0, 1500, num = len(sorted_ids)); 
    for k in range(len(sorted_ids)-1):
        key = str(sorted_ids[k]);
        for idx,df in enumerate(dfs): 
            c = color[idx]
            if key in df.keys():
                x = df.index;
                y = labels[k];
                width = df[key].values;
                
                ax.fill_between(x,y-.8*width,y+.8*width,color=c,alpha = 1)
   
    return 0
