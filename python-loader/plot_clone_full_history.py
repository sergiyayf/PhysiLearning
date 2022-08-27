import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from auxiliary import get_perifery
from auxiliary.analyze_history import * 

matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 10,
                    'font.weight': 'normal',
                    'font.family': 'serif'})
#plt.style.use('default')
#plt.style.use('dark_background')
#custom colors for plotting sectors
custom_blue =(82/255,175/255,230/255)
custom_red =(190/255,28/255,45/255) 

#filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20210524_raven\19_05_raven_switch_at_30_run_9\output\all_data.h5'

for i in range(1,11): 
    filename = r'Z:\Members\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20210426_large_statistics\23_04_run_'+str(i)+r'\output\all_data.h5'
    #filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20210524_raven\19_05_raven_switch_at_30_run_1\output\all_data.h5'
    f = h5py.File(filename, 'r')

    # load history and color 
    history_read = pd.read_hdf(filename,'data/history');
    color_history_read = pd.read_hdf(filename,'data/color_history');

    # get a mask
    masked = mask(history_read, color_history_read)
    # measure width in cell number 
    widths = measure_width(masked)
    print(widths[0])
    # sort by lifetime
    sort(widths)

    # plot 
    cm = 1/2.54
    fig, ax = plt.subplots(figsize=(5*cm,12*cm))
    plot_sorted(ax,sort(widths), widths)

    #ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([1000,0])

    #fig.savefig(r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\no_sw_control\29_11_no_switching_ctrl_run_'+str(i)+r'\output\history.svg',transparent = True)
plt.show()
