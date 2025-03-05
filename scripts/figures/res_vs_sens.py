import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os

def color_plot(df, max_time, ax=None, colormap = plt.cm.viridis):
    if ax is None:
        fig, ax = plt.subplots()
    # plot the number of resistat cells vs total, color the libe with index of the timepoint
    tot = df['Type 0'] + df['Type 1']
    res = df['Type 1']
    res = res[tot > 0]
    tot = tot[tot > 0]

    for i in range(max_time):
        ax.plot(tot[i:i+2], np.log((res[i:i+2])), color=colormap(i/max_time))
    ax.axvline(x=np.mean(tot), color='k', linestyle='--')

os.chdir('/home/saif/Projects/PhysiLearning')

#
fig, ax = plt.subplots()
i=1
j=11
# df = pd.read_hdf(f'./Evaluations/1402_lvs_evals/LvEnvEval__20250206_lv_1_{i}.h5', key=f'run_{j}')
df = pd.read_hdf(f'./Evaluations/1402_pcs_evals/run_{i}.h5', key=f'run_{j}')

colormap = plt.cm.viridis
color_plot(df, 600, ax, colormap)

plt.show()
