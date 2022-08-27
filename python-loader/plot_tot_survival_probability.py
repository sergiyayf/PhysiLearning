import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from auxiliary import get_perifery
from auxiliary.analyze_history import * 
from plot import simulation_clone_history as smh
from scipy import stats
from cycler import cycler
from scipy import interpolate
plt.rcParams.update({'font.size': 10,
                    'font.weight': 'normal',
                    'font.family': 'serif'})

def error(a,b): 
    """
    error of a/b for Poisson
    """
    a = a[1:]
    
    return np.insert(a/b*(np.sqrt( (np.sqrt(a)/a)**2 + (np.sqrt(b)/b)**2)),0,0.)

def get_counts(files):
    """ 
    Get total number of sectors at the perifery by summing them up from all the simulated colonies 
    
    Parameters: 
    ----------
    
    files - list of files to process; 
    
    Returns:
    -------
    
    pandas dataframe with attributes 'total' and 'std', which are total number of clones summed up and their square roots
    """
    file_index = 1 
    count_df = pd.DataFrame()
    rad = [] 
    for filename in files:
        history_read = pd.read_hdf(filename,'data/history');
        history_read = history_read.fillna(value=0.)
        countMatrix = np.array(history_read.values)
        RR = pd.read_hdf(filename,'data/radius')
        
        tot = []
        for i in range(len(countMatrix)):
            (unique, counts) = np.unique(countMatrix[i], return_counts = True)
            
            tot_num = len(unique)-1
            tot.append(tot_num)
        
        interm_df = pd.DataFrame(tot, columns = [str(file_index)]);
        file_index+=1;
        count_df = pd.concat([count_df, interm_df],axis = 1, ignore_index = True)
        rad.append(RR.values) 
    count_df['total'] = count_df.sum(axis = 1)
    count_df['std'] = np.sqrt(count_df['total'].values); 
    
    # make all the simulations of one batch the same length
    length = []
    for element in rad:
        length.append(len(element))
    
    min_sim = min(length)
    cut_R = []
    for Radius in rad:
        
        Radius = Radius[range(min_sim)]
        cut_R.append(Radius)
        
    r = np.mean(np.array(cut_R), axis = 0) 
    return count_df, r


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

def plot_probability(ax,r,data,label='label not specified'):
    """
    plotting survival probabilites, defined in the same way as experimental ones: 
    number of clones now / iniital number of clones
    """
    
    probability = data['total'].values/data['total'].values[0]; 
    err = probability*np.sqrt( (data['std'].values/data['total'].values)**2 +(data['std'].values[0]/data['total'].values[0])**2)
    x = data.index;
    r = r.flatten()
    if len(r) != len(probability): 
        probability=probability[range(len(r))]
        err = err[range(len(r))]
    ax.plot(r, probability, '--', linewidth = 1.0, label = label) 
    ax.fill_between(r, probability-err,probability+err, alpha = 0.5); 
    #plt.legend();
    return 0;

def plot_single_probabilities(ax,files):
    for filename in files:
        history_read = pd.read_hdf(filename,'data/history');
        history_read = history_read.fillna(value=0.)
        countMatrix = np.array(history_read.values)
        RR = pd.read_hdf(filename,'data/radius')
        
        tot = []
        for i in range(len(countMatrix)):
            (unique, counts) = np.unique(countMatrix[i], return_counts = True)
            
            tot_num = len(unique)-1
            tot.append(tot_num)
        
        ax.plot(np.array(RR), tot) 
        ax.set_ylabel('per colony numbers') 
        ax.set_xlabel('Radius')
        
def plot_efficacy(ax,r,no_sw, sw, label = 'efficacy of'): 
    switch = sw['total'].values/sw['total'].values[0]; 
    no_switch = no_sw['total'].values/no_sw['total'].values[0]; 
    r = r.flatten()
    switch_n = sw['total'].values
    no_switch_n = no_sw['total'].values
    # if simulations are not the same size this is to cut things 
    if len(switch) != len (no_switch): 
        smaller = min(len(switch),len(no_switch))
        switch = switch[range(smaller)]
        no_switch = no_switch[range(smaller)] 
        switch_n = switch_n[range(smaller)]
        no_switch_n = no_switch_n[range(smaller)] 
        r = r[range(smaller)]
        
    efficacy = 1 - no_switch/switch
    # efficacy error 
    err = np.sqrt( (error(switch_n,switch_n[0])/switch)**2 + (error(no_switch_n,no_switch_n[0])/no_switch)**2 )
    
    ax.plot(r, efficacy, label = label)
    #ax.fill_between(r,1-no_switch/switch -err,1-no_switch/switch +err,  alpha =0.5 )
    # interpolation to find the cut
    x = r[20:60]
    y = 1-no_switch[20:60]/switch[20:60] - err[20:60]
    f = interpolate.interp1d(x,y)
    g = interpolate.interp1d(x,y+err[20:60])
    
    xnew = np.linspace(x[0],x[1],1000)
    ynew = f(xnew) 
    yline = g(xnew)    
    yintercept = yline[ynew == min(abs(ynew))]
    
    xintercept = xnew[ynew == min(abs(ynew))]
    ax.set_xlim([500,6000])
    ax.set_ylim([-.2,.8])
    
    #ax.axvline(x=xintercept,ymin = 0.2)
    #ax.fill_betweenx([0.,.8],x1=0,x2= xintercept,color=color, alpha = .4)
    ax.axhline(y=0, color='k', linestyle='--')
    #ax.axhline(y=yintercept, color='k', linestyle=':', linewidth = 2, label='Non-zero efficacy')
    
# plot 


colors = ['r', 'c', 'orange', 'g'];

cm = 1/2.54

ncolors = 4
fig, axs = plt.subplots(1, 1, figsize=(8*cm, 6*cm), facecolor='w', edgecolor='k')
axs.set_prop_cycle(cycler('color', plt.get_cmap('viridis', ncolors).colors))

files = []
for k in range(1,21):
    
    filename = r'Z:\Members\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20210426_large_statistics\23_04_run_'+str(k)+'\output\\all_data.h5'
    files.append(filename);

count_df_rand,r = get_counts(files);
plot_probability(axs,r,count_df_rand,'random rescues')

files = []
for k in range(1,11):
    
    filename = r'Z:\Members\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20210810_no_switch_control\04_08_no_switch_control_h5_data\run_'+str(k)+'\\all_data.h5';
    files.append(filename);
    
for k in range(1,11):
    
    filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\no_sw_control\29_11_no_switching_ctrl_run_'+str(k)+'\output\\all_data.h5'
    files.append(filename);

count_df_ns,r = get_counts(files);
plot_probability(axs,r,count_df_ns,'no rescue new')

fig, ax = plt.subplots(1, 1, figsize=(8*cm, 6*cm), facecolor='w', edgecolor='k')
ax.set_prop_cycle(cycler('color', plt.get_cmap('viridis', ncolors).colors))

switches = [30, 35, 40, 50];
for i in range(4):
    mutateAt = switches[i];     
    files = get_files_to_process(mutateAt, 'raven');
    count_df_tr, r = get_counts(files);
    
    plot_probability(axs,r,count_df_tr,'tailored rescues at '+"%i" % r[mutateAt])
    plot_efficacy(ax,r,count_df_ns,count_df_tr,label = 'tailored at %i' % r[mutateAt])
#files = []
#for k in range(1,11):
    
    #filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20211203_neutral_ctrl\03_12_neutral_ctrl_run_'+str(k)+'\output\\all_data.h5'
    #files.append(filename);

#count_df_rand,r = get_counts(files);
#plot_probability(axs,r,count_df_rand,'random rescues')
#fig2, ax2 = plt.subplots(1, 1, figsize=(8*cm, 6*cm), facecolor='w', edgecolor='k')
#ax2.set_prop_cycle(cycler('color', plt.get_cmap('viridis', 10).colors))
#plot_single_probabilities(ax2,files)
#ax2.set_title('random') 

plt.ylabel('Probability')
plt.xlabel('Time, a.u.')
axs.legend() 


plot_efficacy(ax,r,count_df_ns,count_df_rand,label = 'random')
#plot_efficacy(ax,r,count_df_ns,count_df_tr,label = 'tailored at %i' % r[mutateAt])
ax.legend()


#fig2, ax2 = plt.subplots(1, 1, figsize=(8*cm, 6*cm), facecolor='w', edgecolor='k')
#ax2.set_prop_cycle(cycler('color', plt.get_cmap('viridis', 10).colors))
#plot_single_probabilities(ax2,files)
#ax2.set_title('random') 
#fig2.savefig(r'images\random.png')
#plt.savefig('images\probability_sim.svg',transparent=True)
plt.show();
