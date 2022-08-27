import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from auxiliary import get_perifery
from plot import simulation_clone_history as smh
from scipy import stats

# matplotlib font 
import matplotlib 
plt.rcParams.update({'font.size': 24,
                    'font.weight': 'normal'})
#plt.style.use('default')
plt.style.use('dark_background')

#matplotlib.rc('font', **font)

averaged_probability = [] 
averaged_std = [] 
time_when_switched = []

# load files 
for q in [5,10,15,20]:

    files = []
    all_switched = []
    switched_and_survived = []
    for k in range(1,11):
        filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\\20210524_cobra\h5_data_cobra_19_05\\19_05_cobra_switch_at_'+str(q)+'_run_'+str(k)+'\\all_data.h5';
        files.append(filename);
    '''        
    for k in range(1,11):
        for q in [30,35,40,50]:
        
            filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\\20210524_raven\h5_data_folder_raven\\19_05_raven_switch_at_'+str(q)+'_run_'+str(k)+'\\all_data.h5';
            files.append(filename);
    '''  

    for filename in files:
        caused_failure = []
        TitleName = 'run3';
        f = h5py.File(filename, 'r')

        history_read = pd.read_hdf(filename,'data/history');
        history_read = history_read.fillna(value=0.)
        
        #color mask to distinguish types of sectors
        color_history_read = pd.read_hdf(filename,'data/color_history');
        color_mask_type_1 = color_history_read.copy(); 
        color_mask_type_1[color_mask_type_1 == 2.] = 0.
        color_mask_type_1 = color_mask_type_1.fillna(value = 0)

        color_mask_type_2 = color_history_read.copy();
        color_mask_type_2[color_mask_type_2 == 1.] = 0.
        color_mask_type_2[color_mask_type_2 == 2.] = 1.
        color_mask_type_2 = color_mask_type_2.fillna(value = 0) 
    

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
    

        
        for key in df2.keys()[1:]:
            vals = df2[key].values
            inds = np.where(~np.isnan(vals))
            start_day = min(inds[0])
            width = df[key].values[start_day-1]
            
            made_it = ~np.isnan(df2[key].values[q+49])
            caused_failure.append([start_day, made_it,width])
            #print(start_day)
            #print(made_it)
            
        #print(caused_failure)
        occured = np.array(caused_failure)
        time = occured[:,0]
        fail = occured[:,1]
        red_width = occured[:,2]
        survived = time*fail; 
        survived = survived[survived!=0]
        all_switched.append(len(fail))
        switched_and_survived.append(len(survived))
        


    switchedArr = np.array(all_switched)
    survivedArr = np.array(switched_and_survived)

    probabilities = survivedArr/switchedArr; 
    averaged_probability.append(np.mean(probabilities));
    averaged_std.append(np.std(probabilities));
    time_when_switched.append(q); 
    
    print(probabilities)
    print(np.mean(probabilities))
    print(np.std(probabilities))


### Raven the same;


# load files 
for q in [30,35,40,50]:

    files = []
    all_switched = []
    switched_and_survived = []
    for k in range(1,11):
        filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\\20210524_raven\h5_data_folder_raven\\19_05_raven_switch_at_'+str(q)+'_run_'+str(k)+'\\all_data.h5';
        files.append(filename);
    '''        
    for k in range(1,11):
        for q in [30,35,40,50]:
        
            filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\\20210524_raven\h5_data_folder_raven\\19_05_raven_switch_at_'+str(q)+'_run_'+str(k)+'\\all_data.h5';
            files.append(filename);
    '''  

    for filename in files:
        caused_failure = []
        TitleName = 'run3';
        f = h5py.File(filename, 'r')

        history_read = pd.read_hdf(filename,'data/history');
        history_read = history_read.fillna(value=0.)
        
        #color mask to distinguish types of sectors
        color_history_read = pd.read_hdf(filename,'data/color_history');
        color_mask_type_1 = color_history_read.copy(); 
        color_mask_type_1[color_mask_type_1 == 2.] = 0.
        color_mask_type_1 = color_mask_type_1.fillna(value = 0)

        color_mask_type_2 = color_history_read.copy();
        color_mask_type_2[color_mask_type_2 == 1.] = 0.
        color_mask_type_2[color_mask_type_2 == 2.] = 1.
        color_mask_type_2 = color_mask_type_2.fillna(value = 0) 
    

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
    

        
        for key in df2.keys()[1:]:
            vals = df2[key].values
            inds = np.where(~np.isnan(vals))
            start_day = min(inds[0])
            width = df[key].values[start_day-1]
            
            made_it = ~np.isnan(df2[key].values[q+49])
            caused_failure.append([start_day, made_it,width])
            #print(start_day)
            #print(made_it)
            
        #print(caused_failure)
        occured = np.array(caused_failure)
        time = occured[:,0]
        fail = occured[:,1]
        red_width = occured[:,2]
        survived = time*fail; 
        survived = survived[survived!=0]
        all_switched.append(len(fail))
        switched_and_survived.append(len(survived))
        


    switchedArr = np.array(all_switched)
    survivedArr = np.array(switched_and_survived)

    probabilities = survivedArr/switchedArr; 
    averaged_probability.append(np.mean(probabilities));
    averaged_std.append(np.std(probabilities));
    time_when_switched.append(q); 
    
    print(probabilities)
    print(np.mean(probabilities))
    print(np.std(probabilities))



print(time_when_switched)
print(averaged_probability)
print(averaged_std)
plt.figure(figsize=(10,10))
plt.errorbar(time_when_switched, averaged_probability, averaged_std,capsize=20, ls='', color = 'magenta', marker = 'o', linewidth = 4) 
plt.ylim([0,1])
plt.tight_layout()
plt.xlabel('Rescue time')
plt.ylabel('Establishment probability')

plt.show()
