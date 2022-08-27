import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from auxiliary import get_perifery
from plot import simulation_clone_history as smh
from scipy import stats
from auxiliary.analyze_history import * 

# matplotlib font 
import matplotlib 
plt.rcParams.update({'font.size': 24,
                    'font.weight': 'normal'})
#plt.style.use('default')
plt.style.use('dark_background')

#matplotlib.rc('font', **font)

def get_probabilities_and_switching_width(files): 
    """
    get survival / establishment probabilities
    
    Parameters:
    ----------
    
    files - set of files to process, ideally with one switching time
    
    Returns: 
    --------
    
    Establishment probabilities, width 
    """
    # loop through all the files 
    number_of_all_switched, number_of_switched_and_survived, number_of_red_cells_at_the_switching_timepoint = [],[],[]
    for filename in files:
        
        # initiated some lists to store data there 
        caused_failure = []
        
        # read data 
        f = h5py.File(filename, 'r')
        history_read = pd.read_hdf(filename,'data/history');
        history_read = history_read.fillna(value=0.)
        #color mask to distinguish types of sectors
        color_history_read = pd.read_hdf(filename,'data/color_history');
        # get a mask
        masked = mask(history_read, color_history_read)
        # measure width in cell number 
        widths = measure_width(masked)
        df2 = widths[1]
        df = widths[0]
        print('file')
        # go through all blue clones, ignore the one with index 0, it is an artifact
        for key in df2.keys()[1:]:
            # get the width values 
            vals = df2[key].values
            # get the times where clones were alived
            inds = np.where(~np.isnan(vals))
            # get switching day of a clone, for controlled switching this is not really necessary 
            start_day = min(inds[0])
            # get the width of red clone for which switch occured
            width = df[key].values[start_day-1]
            # for this clone find out if it made it thorugh 49 timepoints or not 
            made_it = ~np.isnan(df2[key].values[start_day+45])
            # save it in the list [day when switched, if made it or not, widht of red]
            caused_failure.append([start_day, made_it,width])
            
            
        clone_statistics = np.array(caused_failure)
        switching_time = clone_statistics[:,0]
        survived_or_died = clone_statistics[:,1]
        red_width_at_switching_timepoint = clone_statistics[:,2]
        
        number_of_all_switched.append(len(survived_or_died))
        number_of_switched_and_survived.append(len(survived_or_died[survived_or_died==1.]))
        number_of_red_cells_at_the_switching_timepoint.append(np.nansum(red_width_at_switching_timepoint))
        
    return number_of_all_switched, number_of_switched_and_survived, number_of_red_cells_at_the_switching_timepoint 

def error(a,b): 
    """
    error of a/b for Poisson
    """
    return a/b*(np.sqrt( (np.sqrt(a)/a)**2 + (np.sqrt(b)/b)**2))

def save_to_h5(time, files, cluster): 
    n_switched, n_survived, red_width = get_probabilities_and_switching_width(files)

    tot_switched = np.sum(n_switched)
    survivals = np.sum(n_survived) 
    red_cells = np.sum(red_width)
    
    h5File ='survivals.h5'
    d = {'tot_switched': [tot_switched], 'survivals':[survivals], 'red_cells': [red_cells], 'switching_time': [time]}
    df = pd.DataFrame.from_dict(d)
    print(df)
    df.to_hdf(h5File,'data_'+cluster+str(time))
    return 0 
def plot_wrapper(time,files,color,lbl):
    """
    plot
    
    """
    
    n_switched, n_survived, red_width = get_probabilities_and_switching_width(files)

    tot_switched = np.sum(n_switched)
    survivals = np.sum(n_survived) 
    red_cells = np.sum(red_width)



    
    plt.errorbar(time, survivals/tot_switched, error(survivals,tot_switched),capsize=20, ls='', color = color, marker = 'o', linewidth = 4,label = lbl) 
    #plt.ylim([0,1])
    plt.tight_layout()
    plt.xlabel('Rescue time')
    plt.ylabel('Establishment probability')
    return 0
    
# load files 
plt.figure(figsize=(10,10))
files = []
  
for k in range(1,11):
    filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20211014_raven_switch_at_45\14_10_2021_raven_switch_at_45_run_'+str(k)+r'\output\all_data.h5'
    files.append(filename);
    
plot_wrapper(45,files,'green','raven other') 
"""
save_to_h5(45,files,'raven_other1') 
files = []
  
for k in range(1,11):
    filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20211014_raven_switch_at_20\14_10_2021_raven_switch_at_20_run_'+str(k)+r'\output\all_data.h5'
    files.append(filename);
    
plot_wrapper(20,files,'c','raven other') 
save_to_h5(20,files,'raven_other2') 
for q in [5,10,15,20]:

    files = []
  
    for k in range(1,11):
        filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\\20210524_cobra\h5_data_cobra_19_05\\19_05_cobra_switch_at_'+str(q)+'_run_'+str(k)+'\\all_data.h5';
        files.append(filename);
        
    plot_wrapper(q,files,'magenta','cobra')
    save_to_h5(q,files,'cobra') 

for q in [30,35,40,50]:

    files = []
  
    for k in range(1,11):
        filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\\20210524_raven\h5_data_folder_raven\\19_05_raven_switch_at_'+str(q)+'_run_'+str(k)+'\\all_data.h5';
        files.append(filename);
        
    plot_wrapper(q,files,'blue','raven')
    save_to_h5(q,files,'raven') 

   
"""    
plt.legend()
plt.show()
