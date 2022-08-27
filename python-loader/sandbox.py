from load.h5tohist import *
import pandas as pd
from auxiliary.analyze_history import *
import matplotlib as mpl
#mpl.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
plt.rcParams.update({'font.size': 8,
                    'font.weight': 'normal',
                    'font.family': 'serif'})
mpl.rcParams['pdf.fonttype'] = 42 #to make text editable in pdf output
mpl.rcParams['font.sans-serif'] = ['Arial'] #to make it Arial
def error(a,b): 
    """
    error of a/b for Poisson
    """
        
    return a/b*(np.sqrt( (np.sqrt(a)/a)**2 + (np.sqrt(b)/b)**2))

h5File = 'ER_data_collection.h5'
f = h5py.File(h5File, 'r')

for i  in range(14,15):
    df = pd.read_hdf(h5File, 'data/random_switching_new/ds'+str(i)+'/red')

    df2 = pd.read_hdf(h5File, 'data/random_switching_new/ds'+str(i)+'/blue')
    width = [df,df2]

    cm = 1/2.54
    fig, ax = plt.subplots(figsize=(5*cm,12*cm))
    plot_sorted(ax,sort(width), width)
    ax.set_yticks([])
    ax.set_xticks([])

#fig.savefig('images/random_switch_trajectory.pdf',transparent = True)
df = pd.read_hdf(h5File, 'data/tailored_switch_at_30/ds3/red')

df2 = pd.read_hdf(h5File, 'data/tailored_switch_at_30/ds3/blue')
width = [df,df2]

cm = 1/2.54
fig, ax = plt.subplots(figsize=(5*cm,12*cm))
plot_sorted(ax,sort(width), width)
ax.set_yticks([])
ax.set_xticks([])
#fig.savefig('images/tailored_switch_at_30_trajectory.pdf',transparent = True)


ncolors=9
fig, ax = plt.subplots(figsize=(5*cm,12*cm))
ax.set_prop_cycle(cycler('color', plt.get_cmap('viridis', ncolors).colors) *
                   cycler('linestyle', ['-', '--']))

for switch in [5,15,20,25,30,35,40,45,50]:
    counts_red = 0 
    counts_blue= 0
    counts_tot = 0
    for key in f['data/tailored_switch_at_'+str(switch)]:
        
        df = pd.read_hdf(h5File, 'data/tailored_switch_at_'+str(switch)+'/'+key+'/red')
        df2 = pd.read_hdf(h5File, 'data/tailored_switch_at_'+str(switch)+'/'+key+'/blue')
        tot = df.add(df2, axis=1,fill_value=0)
        
        counts_red += df.count(axis=1).values 
        counts_blue += df2.count(axis=1).values
        counts_tot += tot.count(axis=1).values
    #ax.plot(df.index,counts_red,label='red')
    probability = counts_tot/counts_tot[0]
    err = error(counts_tot,counts_tot[0])
    #ax.plot(df.index,counts_tot/counts_tot[0],label='tot_'+str(df.index[switch]))
    ax.fill_between(df.index,probability-err,probability+err,label='tot_'+str(df.index[switch]),alpha = 0.5)
    ax.plot(df.index,counts_blue/counts_tot[0],label='blue_'+str(df.index[switch]))

counts_red = 0 
counts_blue= 0
counts_tot = 0
fig3,ax3 = plt.subplots()
for key in f['data/random_switching_new']:
    
    df = pd.read_hdf(h5File, 'data/random_switching_new/'+key+'/red')
    df2 = pd.read_hdf(h5File, 'data/random_switching_new/'+key+'/blue')
    tot = df.add(df2, axis=1,fill_value=0)
    
    counts_red += df.count(axis=1).values 
    counts_blue += df2.count(axis=1).values
    counts_tot += tot.count(axis=1).values
#ax.plot(df.index,counts_red,label='red')
probability = counts_tot/counts_tot[0]
err = error(counts_tot,counts_tot[0])
#ax.plot(df.index,counts_tot/counts_tot[0],label='tot_'+str(df.index[switch]))
ax3.fill_between(df.index,probability-err,probability+err,label='rs',alpha = 0.5,color = 'r')
ax3.plot(df.index,counts_blue/counts_tot[0],ls = '--',label='blue_rs')

counts_red = 0 
counts_blue= 0
counts_tot = 0
fig2,ax2 = plt.subplots()
for key in f['data/no_switch_control']:
    if key != 'ds1' and key!='ds2' and key != 'ds9' and key != 'ds4' and key!= 'ds17' and key!='ds14': 
        df = pd.read_hdf(h5File, 'data/no_switch_control/'+key+'/red')
            
        counts_red += df.count(axis=1).values 
        ax2.plot(df.index,df.count(axis=1).values,label= key)
   
probability = counts_red/counts_red[0]
err = error(counts_red,counts_red[0])
ax.plot(df.index,counts_red/counts_red[0],color = 'k',label='no sw')
ax3.fill_between(df.index,probability-err,probability+err,label='no s',alpha = 0.5,color = 'k')
ax3.plot(df.index,counts_red/counts_red[0],color = 'k',label='no sw')
ax2.legend()
ax3.legend()
fig3.savefig('images/random_p_sruv_placeholder.pdf',transparent = True)
ax3.set_yscale('log') 
ax3.set_xscale('log')
 
#ax.legend()

plt.show()
