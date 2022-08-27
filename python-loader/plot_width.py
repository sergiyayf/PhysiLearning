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
plt.rcParams.update({'font.size': 12,
                    'font.weight': 'normal'})
#plt.style.use('default')
plt.style.use('dark_background')

#matplotlib.rc('font', **font)

def get_averaged_width(files): 
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
    
    r_w, b_w, R = [],[],[]
    for filename in files:
        
        # initiated some lists to store data there 
                
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
        
        df = df.drop(columns=['0.0'],axis=1)
        df2 = df2.drop(columns=['0.0'],axis=1)
        df['mean'] = df.mean(axis=1)
        df2['mean'] = df2.sum(axis=1)/len(df2.keys())
        
        #df2['mean'] = df2.mean(axis=1)
        df['std'] = df.std(axis=1)
        df2['std'] = df2.std(axis=1)
                
        
        r_w.append(df['mean'].values)
        
        b_w.append(df2['mean'].values)
        RR = pd.read_hdf(filename,'data/radius')
        
        R.append(RR['Rad'].values)
        
    
    # make all the simulations of one batch the same length
    length = []
    for element in r_w:
        length.append(len(element))
    
    min_sim = min(length)
    cut_red,cut_blue,cut_R = [],[],[]
    for red, blue, Radius in zip(r_w,b_w,R):
        red = red[range(min_sim)]
        blue = blue[range(min_sim)]
        Radius = Radius[range(min_sim)]
        cut_red.append(red)
        cut_blue.append(blue)
        cut_R.append(Radius)
        
    red_w = np.nanmean(np.array(cut_red),axis = 0)
    red_std = np.nanstd(np.array(cut_red),axis = 0)
    #blue_w = np.nanmean(np.array(cut_blue),axis = 0) 
    blue_w = np.nansum(np.array(cut_blue),axis = 0)/ len(cut_blue)
    
    blue_std = np.nanstd(np.array(cut_blue),axis = 0) 
    rad = np.nanmean(np.array(cut_R),axis = 0)
    rad_std = np.nanstd(np.array(cut_R),axis = 0)
    #rad = range(len(red_w))
    blue_w[blue_w==0]=np.nan
    return red_w, blue_w, rad, red_std, blue_std, rad_std 

def plot(files): 
    red_w, blue_w ,rad, red_std, blue_std, rad_std = get_averaged_width(files) 
    #ax1.plot(rad,red_w,color = 'r',label = 'red')
    ax1.plot(rad,blue_w, color = 'c', label = 'blue')
    #plt.fill_between(rad,red_w-red_std, red_w+red_std, color = 'r', alpha = 0.5) 
    #plt.fill_between(rad, blue_w-blue_std, blue_w+blue_std, color ='c', alpha =0.5) 
    

def fit_line(x,y): 
    new_x = x[~np.isnan(y)]
    new_y = y[~np.isnan(y)] 
   
    coef = np.polyfit(new_x,new_y,1) 
    poly1d_fn = np.poly1d(coef) 
    # poly1d_fn is now a function which takes in x and returns an estimate for y
    
    slope = 1000*coef[0]
    ax1.plot(new_x, poly1d_fn(new_x), '--w')
    ax1.text(new_x[-1],poly1d_fn(new_x[-1]),'%.1f'% slope, color = 'w')
    ax2.scatter(new_x[0],coef[0]*(new_x[0]))
    print(coef[0]/new_x[0])
    print(new_x[0])
    print(coef[0])
    print('next')
    return 0
    
fig, (ax1,ax2) = plt.subplots(1,2)    
files = []
  
for k in range(1,11):
    filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20211014_raven_switch_at_45\14_10_2021_raven_switch_at_45_run_'+str(k)+r'\output\all_data.h5'
    files.append(filename);
    
plot(files) 
red_w, blue_w ,rad, red_std, blue_std, rad_std = get_averaged_width(files) 
fit_line(rad,blue_w)
files = []
for k in range(1,11):
    filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\\20210524_cobra\h5_data_cobra_19_05\\19_05_cobra_switch_at_'+str(5)+'_run_'+str(k)+'\\all_data.h5';
    files.append(filename);
    
plot(files)
red_w, blue_w ,rad, red_std, blue_std, rad_std = get_averaged_width(files) 
fit_line(rad,blue_w)
files = []
for k in range(1,11):
    filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\\20210524_cobra\h5_data_cobra_19_05\\19_05_cobra_switch_at_'+str(10)+'_run_'+str(k)+'\\all_data.h5';
    files.append(filename);
    
plot(files) 
red_w, blue_w ,rad, red_std, blue_std, rad_std = get_averaged_width(files) 
fit_line(rad,blue_w)
files = []
for k in range(1,11):
    filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\\20210524_cobra\h5_data_cobra_19_05\\19_05_cobra_switch_at_'+str(15)+'_run_'+str(k)+'\\all_data.h5';
    files.append(filename);
    
plot(files) 
red_w, blue_w ,rad, red_std, blue_std, rad_std = get_averaged_width(files) 
fit_line(rad,blue_w)
files = []
for k in range(1,11):
    filename = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\\20210524_cobra\h5_data_cobra_19_05\\19_05_cobra_switch_at_'+str(20)+'_run_'+str(k)+'\\all_data.h5';
    files.append(filename);
    
plot(files) 
red_w, blue_w ,rad, red_std, blue_std, rad_std = get_averaged_width(files) 
fit_line(rad,blue_w)
files = []
for k in range(1,11):
    filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20210524_raven\h5_data_folder_raven\19_05_raven_switch_at_'+str(30)+'_run_'+str(k)+'\\all_data.h5';
    files.append(filename);
    
plot(files) 
red_w, blue_w ,rad, red_std, blue_std, rad_std = get_averaged_width(files) 
fit_line(rad,blue_w)

files = []
for k in range(1,11):
    filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20210524_raven\h5_data_folder_raven\19_05_raven_switch_at_'+str(35)+'_run_'+str(k)+'\\all_data.h5';
    files.append(filename);
    
plot(files) 
red_w, blue_w ,rad, red_std, blue_std, rad_std = get_averaged_width(files) 
fit_line(rad,blue_w)

files = []
for k in range(1,11):
    filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20210524_raven\h5_data_folder_raven\19_05_raven_switch_at_'+str(40)+'_run_'+str(k)+'\\all_data.h5';
    files.append(filename);
    
plot(files) 
red_w, blue_w ,rad, red_std, blue_std, rad_std = get_averaged_width(files) 
fit_line(rad,blue_w)

files = []
for k in range(1,11):
    filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20210524_raven\h5_data_folder_raven\19_05_raven_switch_at_'+str(50)+'_run_'+str(k)+'\\all_data.h5';
    files.append(filename);
    
plot(files) 
red_w, blue_w ,rad, red_std, blue_std, rad_std = get_averaged_width(files) 
fit_line(rad,blue_w)

ax1.set_ylim([0,12])
ax1.set_title('Average clone width') 
ax1.set_xlabel('Radius')
ax1.set_ylabel('Mean clone width')

ax2.set_ylim([0,.2])
ax2.set_title(r'$\frac{dw}{dR} (R*)$') 
ax2.set_ylabel(r'slope$\cdot \sqrt{R*}$')
ax2.set_xlabel('R*') 
#plt.legend()

plt.show()
