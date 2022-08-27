import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from auxiliary import get_perifery
from plot import simulation_clone_history as smh
from scipy import stats

import matplotlib 
plt.rcParams.update({'font.size': 22,
                    'font.weight': 'normal'})
plt.style.use('dark_background')
def get_counts_structure(files): 
    file_index = 1;
       
    count_df = pd.DataFrame()
    for filename in files:
        #filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\\20210426_large_statistics\\23_04_run_'+str(q)+'\output\\all_data.h5';
        #filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\\20210517_switch_control\\17_05_switch_at_'+str(q)+'0_1\output\\all_data.h5';
        TitleName = 'run3';
        f = h5py.File(filename, 'r')

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
        #print(tp1)
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
        
        #sorting 
        
        lengths2 = []
        keys2 = []
        for key in df2.keys(): 
            x = df2.index
            y = float(key)*np.ones(len(x))
            width = df2[key].values
            length = np.sum(~np.isnan(width))
            lengths2.append(length)
            keys2.append(float(key))
                
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
            
        df_sort = pd.DataFrame(lengths)
        df_sort = df_sort.T
        df_sort.columns = keys
            
        combined_df = df_sort.add(df_sort2, fill_value=0)
        combined_df = combined_df.sort_values(by=0, axis = 1)
        #plt.ylim([3*7500,3*7800])  
            
        countMatrix = np.array(history_read.values)
        
        
        tot = []
        for i in range(len(countMatrix)):
            (unique, counts) = np.unique(countMatrix[i], return_counts = True)
            
            tot_num = len(unique)-1
            tot.append(tot_num)
        
        interm_df = pd.DataFrame(tot, columns = [str(file_index)]);
        file_index+=1;
        count_df = pd.concat([count_df, interm_df],axis = 1, ignore_index = True)
        

    count_df['total'] = count_df.sum(axis = 1)
    count_df['std'] = np.sqrt(count_df['total'].values); 
    return count_df;

def get_files_to_process(time_when_switch=30, cluster='raven'):
    """
    cluster 1  - raven
    cluster 2 - cobra
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

def plot_probability(data,color='k',label='label not specified'):
    probability = data['total'].values/data['total'].values[0]; 
    err = probability*np.sqrt( (data['std'].values/data['total'].values)**2 +(data['std'].values[0]/data['total'].values[0])**2)
    x = data.index;
    print(probability)
    plt.plot(x, probability, '--', color = color, linewidth = 1.0, label = label) 
    plt.fill_between(x, probability-err,probability+err, color = color, alpha = 0.5); 
    plt.legend();
    return 0;

# plot 
switches = [30, 35, 40, 50];
colors = ['r', 'c', 'orange', 'g'];
plt.figure(figsize=(10,10))
plt.rcParams.update({'font.size': 26})
for i in range(1):
    mutateAt = switches[i];     
    files = get_files_to_process(mutateAt, 'raven');
    count_df = get_counts_structure(files);
    plot_probability(count_df,'r','tailored rescues at '+str(mutateAt))


switches = [5, 10, 15, 20];
colors = ['m', 'k', 'grey', 'y'];    
files = [];
for k in range(1,11):
    
    filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\\20210810_no_switch_control\\04_08_no_switch_control_h5_data\\run_'+str(k)+'\\all_data.h5';
    files.append(filename);

count_df = get_counts_structure(files);
plot_probability(count_df,'w','no rescue')
i+=2;
for k in range(1,21):
    
    filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20210426_large_statistics\23_04_run_'+str(k)+'\output\\all_data.h5'
    files.append(filename);

count_df = get_counts_structure(files);
plot_probability(count_df,'orange','random rescues')
plt.xlim([0,89])
plt.ylabel('Probability',fontsize=26)
plt.xlabel('Time, a.u.', fontsize = 26)
plt.savefig('probability_sim.png')
plt.show();


