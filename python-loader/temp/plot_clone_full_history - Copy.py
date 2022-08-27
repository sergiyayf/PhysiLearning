import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from auxiliary import get_perifery
plt.rcParams.update({'font.size': 24,
                    'font.weight': 'normal'})
#plt.style.use('default')
plt.style.use('dark_background')

# Choose a colony for which clone history would be plot
#filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20210524_cobra\h5_data_cobra_19_05\19_05_cobra_switch_at_10_run_7\all_data.h5'; 
filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20210507_control_simulations\07_05_control_3\output\all_data.h5'
#filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\20210524_raven\h5_data_folder_raven\19_05_raven_switch_at_50_run_2\all_data.h5';
#filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\20210426_large_statistics\23_04_run_1\output\all_data.h5';

f = h5py.File(filename, 'r')

cyan_cell_type = 2; 
def plot_history(filename):
    history_read = pd.read_hdf(filename,'data/history');
    history_read = history_read.fillna(value=0.)
    #print(history_read)
    color_history_read = pd.read_hdf(filename,'data/color_history');
    color_mask_type_1 = color_history_read.copy(); 
    color_mask_type_1[color_mask_type_1 == 2.] = 0.
    color_mask_type_1 = color_mask_type_1.fillna(value = 0)

    color_mask_type_2 = color_history_read.copy();
    color_mask_type_2[color_mask_type_2 == 1.] = 0.
    color_mask_type_2[color_mask_type_2 == 2.] = 1.
    color_mask_type_2 = color_mask_type_2.fillna(value = 0) 
    #print(color_mask_type_1)
    #print(color_history_read)

    tp1 = history_read * color_mask_type_1.values
    tp2 = history_read * color_mask_type_2.values

    tp1_array = np.array(tp1) 
    
    # count width by the cells with the same iD 
    df = pd.DataFrame() 
    for i in range(len(tp1_array)):
        (unique, counts) = np.unique(tp1_array[i], return_counts = True)
        #print(unique)
        unique_str = [str(unique[i]) for i in range(len(unique))]
        #print(unique_str)
        dictionary = {unique_str[i] : [counts[i]] for i in range(len(counts))}
    # print(dictionary)
        data = pd.DataFrame.from_dict(dictionary) 
        
        #print(data)
        df=pd.concat([df,data],axis = 0, ignore_index = True)

    #df = df.fillna(value = 0.)

    tp2_array = np.array(tp2) 
    #print(tp2)
    df2 = pd.DataFrame() 
    for i in range(len(tp2_array)):
        (unique, counts) = np.unique(tp2_array[i], return_counts = True)
        #print(unique)
        unique_str = [str(unique[i]) for i in range(len(unique))]
        #print(unique_str)
        dictionary = {unique_str[i] : [counts[i]] for i in range(len(counts))}
    # print(dictionary)
        data = pd.DataFrame.from_dict(dictionary) 
        
        #print(data)
        df2=pd.concat([df2,data],axis = 0, ignore_index = True)

    #df2 = df2.fillna(value = 0.)
    
    #sorting 
    plt.figure()
    lengths2 = []
    keys2 = []
    for key in df2.keys(): 
        x = df2.index
        y = float(key)*np.ones(len(x))
        width = df2[key].values
        length = np.sum(~np.isnan(width))
        lengths2.append(length)
        keys2.append(float(key))
        plt.fill_between(x,y*3-width,y*3+width,color='c',alpha = 0.5)        
    df_sort2 = pd.DataFrame(lengths2)
    df_sort2 = df_sort2.T
    df_sort2.columns = keys2
        
    lengths = []
    keys = []
    for key in df.keys(): 
        x = df.index
        y = float(key)*np.ones(len(x))
        width = df[key].values
        length = np.sum(~np.isnan(width))
        lengths.append(length)
        keys.append(float(key))
        plt.fill_between(x,y*3-width,y*3+width,color='r',alpha = 0.5)
    df_sort = pd.DataFrame(lengths)
    df_sort = df_sort.T
    df_sort.columns = keys
        
    combined_df = df_sort.add(df_sort2, fill_value=0)
    combined_df = combined_df.sort_values(by=0, axis = 1)
    plt.ylim([3*7500,3*7800])  
    
    # plotting sorted 
     
    plt.figure(figsize=(8,10))
    
    labels = np.linspace(0, 1000, num = len(combined_df.keys())); 
    for k in range(len(combined_df.keys())-1):
        key = str(combined_df.keys()[k]);
        #print(key)
        if key in df.keys():
            x = df.index;
            y = labels[k];
            width = df[key].values;
           
            plt.fill_between(x,y-.8*width,y+.8*width,color='r',alpha = 1)
            if key in df2.keys():
                x = df2.index;
                y = labels[k];
                width = df2[key].values;
                plt.fill_between(x,y-.8*width,y+.8*width,color='c',alpha = 1)
    
    plt.ylim([1000,0])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Time')
    plt.ylabel('Clone')
    plt.tight_layout()
    #plt.savefig('fig.png')

    
    countMatrix = np.array(history_read.values)
    
    
    tot = []
    for i in range(len(countMatrix)):
        (unique, counts) = np.unique(countMatrix[i], return_counts = True)
        
        tot_num = len(unique)-1
        tot.append(tot_num)
    plt.figure(figsize=(10,10))
    plt.plot(tot,label = 'Total number',linewidth = 4.0)
    plt.xlabel('Time')
    plt.ylabel('Number')
    plt.legend()
    
    
    
plot_history(filename)
plt.show()
