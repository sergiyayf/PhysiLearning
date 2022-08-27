import h5py
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from auxiliary import get_perifery
from plot import simulation_clone_history as smh
from scipy import stats
from auxiliary.analyze_history import * 

# matplotlib font 

plt.rcParams.update({'font.size': 10,
                    'font.weight': 'normal',
                    'font.family': 'serif'})
#plt.style.use('default')
#plt.style.use('dark_background')

def mean(key,time):
    r_w, b_w, R = [],[],[]
    for scene in range(10): 
        df = pd.read_hdf(filename, 'data/t'+str(time)+'/red/c'+str(scene))
        
        df2 = pd.read_hdf(filename, 'data/t'+str(time)+'/blue/c'+str(scene))
        RR = df['rad']
        
        df = df.drop(columns=['0.0','rad','switching_time'],axis=1)
        df2 = df2.drop(columns=['0.0','rad','switching_time'],axis=1) 
        d = df2.tail(1).iloc[0].isna()
        d = pd.DataFrame(d)
        #print(d)
        #print(d.keys())
        
        select_indices = list(np.where(d[d.keys()[0]] == False)[0])
        
        #print(select_indices)
        #print(d.iloc[select_indices])
        dd = d.iloc[select_indices]
        #print(dd.index.values.tolist())
        key_names = dd.index.values.tolist()
        new_df2 = df2[key_names]
        df2 = new_df2
        #print('scene',scene)
        #print(df2)
        #print(new_df2)
        df['mean'] = df.mean(axis=1)
        df2['mean'] = df2.mean(axis=1)
        
        #df2['mean'] = df2.mean(axis=1)
        df['std'] = df.std(axis=1)
        df2['std'] = df2.std(axis=1)
        #print(df2)        
        
        r_w.append(df['mean'].values)
        
        b_w.append(df2['mean'].values)
                    
        R.append(RR.values)
        
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
    blue_w = np.nanmean(np.array(cut_blue),axis = 0)
    
    blue_std = np.nanstd(np.array(cut_blue),axis = 0) 
    rad = np.nanmean(np.array(cut_R),axis = 0)
    rad_std = np.nanstd(np.array(cut_R),axis = 0)
    #rad = range(len(red_w))
    blue_w[blue_w==0]=np.nan
    
    return red_w, blue_w, rad, red_std, blue_std, rad_std 

def plot(files): 
    red_w, blue_w ,rad, red_std, blue_std, rad_std = get_averaged_width(files) 
    ax1.plot(rad,red_w,color = 'r',label = 'red')
    ax1.plot(rad,blue_w, color = 'c', label = 'blue')
    plt.fill_between(rad,red_w-red_std, red_w+red_std, color = 'r', alpha = 0.5) 
    plt.fill_between(rad, blue_w-blue_std, blue_w+blue_std, color ='c', alpha =0.5) 

def save_to_h5(widths, index, time): 
    h5File ='widths.h5'
    
    df2 = widths[1]
    df = widths[0]
    RR = pd.read_hdf(filename,'data/radius')
    df['rad'] = RR.values
    df2['rad'] = RR.values 
    df['switching_time'] = time 
    df2['switching_time'] = time 
    df.to_hdf(h5File,'data/t'+str(time)+'/red/c'+str(index))
    df2.to_hdf(h5File,'data/t'+str(time)+'/blue/c'+str(index))
    print(index)
    return 0     

def fit_line(x,y): 
    new_x = x[~np.isnan(y)]
    new_y = y[~np.isnan(y)] 
   
    coef = np.polyfit(new_x,new_y,1) 
    poly1d_fn = np.poly1d(coef) 
    # poly1d_fn is now a function which takes in x and returns an estimate for y
    
    slope = 1000*coef[0]
    ax1.plot(new_x, poly1d_fn(new_x), '--k')
    ax1.text(new_x[-1],poly1d_fn(new_x[-1]),'%.1f'% slope, color = 'k')
    #ax2.scatter(new_x[0],coef[0]*np.sqrt(new_x[0]))
    #print(coef[0]/new_x[0])
    #print(new_x[0])
    #print(coef[0])
    #print('next')
    return 0
 
# main
#fig, (ax1,ax2) = plt.subplots(1,2) 
cm = 1/2.54
fig, ax1 = plt.subplots(figsize=(12*cm,8*cm))
fig2, ax2 = plt.subplots(figsize=(12*cm,8*cm))
custom_blue =(82/255,175/255,230/255)
custom_red =(190/255,28/255,45/255)
    
filename = r'data\widths.h5'; 
f = h5py.File(filename, 'r')
print(f['data/t15/red/'].keys())
for time in [5,10,15,20,30,35,40,45,50]:
    for key in f['data/t'+str(time)].keys():
        
        red_w, blue_w ,rad, red_std, blue_std, rad_std = mean(key,time) 
        ax1.plot(rad,blue_w, color = custom_blue, label = 'blue')
        ax1.plot(rad,red_w, color = custom_red, label = 'red')
        ax2.errorbar(rad,blue_w,yerr=blue_std, color = custom_blue, label = 'blue')
        ax2.errorbar(rad,red_w,yerr =red_std, color = custom_red, label = 'red')
        fit_line(rad,blue_w)
            
ax1.set_xlabel(r'R, $\mu m$')
ax1.xaxis.set_label_coords(1., -0.025)
ax1.set_ylabel('Width, NoC', rotation=0) 
ax1.yaxis.set_label_coords(0.05, 1.0)
ax1.set_ylim(ymax=13.99)
ax1.set_xlim(xmax=6100)
ax1.set_xticks([1000,2000,3000,4000,5000])
#ax1.set_title('Clone width development')
#plt.tight_layout()
#fig.savefig('images\width.svg', transparent=True)

plt.show()
