import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from auxiliary import get_perifery
from plot import simulation_clone_history as smh
from scipy import stats

import matplotlib 
plt.rcParams.update({'font.size': 24,
                    'font.weight': 'normal'})
#plt.style.use('default')
plt.style.use('dark_background')

#matplotlib.rc('font', **font)
plt.style.use('dark_background')
caused_failure = []
files = []
for k in range(1,11):
    for q in [5,10,15]:
    #filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\\20210426_large_statistics\\23_04_run_'+str(q)+'\output\\all_data.h5';
        filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\\20210524_cobra\h5_data_cobra_19_05\\19_05_cobra_switch_at_'+str(q)+'_run_'+str(k)+'\\all_data.h5';
        files.append(filename);
        
for k in range(1,11):
    for q in [30,35,40,50]:
    #filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\\20210426_large_statistics\\23_04_run_'+str(q)+'\output\\all_data.h5';
        filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\\20210524_raven\h5_data_folder_raven\\19_05_raven_switch_at_'+str(q)+'_run_'+str(k)+'\\all_data.h5';
        files.append(filename);
#for k in range(1,11):
    
    #filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\20210617_switch_at_25\17_06_raven_switch_at_25_run_'+str(k)+r'\output\all_data.h5';
    #files.append(filename); 
for k in range(1,11):
    
    filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20211014_raven_switch_at_45\14_10_2021_raven_switch_at_45_run_'+str(k)+r'\output\all_data.h5';
    files.append(filename); 
    
for k in range(1,11):
    
    filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20211014_raven_switch_at_20\14_10_2021_raven_switch_at_20_run_'+str(k)+r'\output\all_data.h5';
    files.append(filename); 

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
 

    
    for key in df2.keys()[1:]:
        vals = df2[key].values
        inds = np.where(~np.isnan(vals))
        start_day = min(inds[0])
        width = df[key].values[start_day-1]
        
        made_it = ~np.isnan(df2[key].values[-1])
        caused_failure.append([start_day, made_it,width])
        #print(start_day)
        #print(made_it)
occured = np.array(caused_failure)
time = occured[:,0]
fail = occured[:,1]
red_width = occured[:,2]
survived = time*fail; 
survived = survived[survived!=0]

## survived with width 
print(red_width)
print(fail)
width_surv = red_width*fail;
width_surv = width_surv[width_surv!=0]
print(width_surv)
(unique_w, counts_w) = np.unique(width_surv, return_counts = True)

print(unique_w)
print(counts_w)

(unique_occ, counts_occ) = np.unique(time, return_counts = True)
(unique_surv, counts_surv) = np.unique(survived, return_counts = True)


occ_bin_sum, occ_bin_edges, occ_binnnumber = stats.binned_statistic( unique_occ, counts_occ, statistic='sum',  bins = 5) 
surv_bin_sum, surv_bin_edges, surv_binnnumber = stats.binned_statistic( unique_surv, counts_surv, statistic='sum',  bins = 5) 


plt.figure()
plt.hlines(occ_bin_sum, occ_bin_edges[:-1], occ_bin_edges[1:], colors='b', lw=2, label='occ')
#plt.errorbar((occ_bin_edges[:-1]+occ_bin_edges[1:])/2, occ_bin_sum, np.sqrt(occ_bin_sum), label = 'std, Poisson')
plt.hlines(surv_bin_sum,surv_bin_edges[:-1],surv_bin_edges[1:], colors='orange', lw=2, label = 'surv')
plt.legend()

plt.figure(figsize=(8,8))

plt.hlines(surv_bin_sum/occ_bin_sum,surv_bin_edges[:-1],surv_bin_edges[1:], linewidth = 4, linestyle = '--', colors='black', lw=2, label = 'relative')
plt.vlines ((surv_bin_edges[:-1]+surv_bin_edges[1:])/2, 0,surv_bin_sum/occ_bin_sum)
#plt.annotate("",
            #xy=((surv_bin_edges[5]+surv_bin_edges[5])/2, surv_bin_sum[5]/occ_bin_sum[5]), xycoords='data',
            #xytext=((surv_bin_edges[5]+surv_bin_edges[5])/2, surv_bin_sum[3]/occ_bin_sum[3]), textcoords='data',
            #arrowprops=dict(arrowstyle="<->",
                            #connectionstyle="arc3", color='r', lw=2),
            #)
#plt.text((surv_bin_edges[5]+surv_bin_edges[5])/2,0.3, '%.2f' %(surv_bin_sum[3]/occ_bin_sum[3]-surv_bin_sum[5]/occ_bin_sum[5]) , 
         #rotation = 90, fontsize = 16)
plt.ylim([0,1]) 
plt.xlim([0,100])
plt.ylabel('Establishment probability') 
plt.xlabel('Time window')
#plt.legend()

plt.figure()
plt.hist(time,bins = 5, label = 'all switched')
plt.title('occured')

plt.hist(survived,bins=5, color = 'orange', label = 'switched and survived')
plt.title('survived')
plt.xlabel('Time window when switched') 
plt.ylabel('Total number out of 20 simulations')
plt.legend()


### plot width of survived
plt.figure()
plt.plot(unique_w, counts_w)

plt.show()
