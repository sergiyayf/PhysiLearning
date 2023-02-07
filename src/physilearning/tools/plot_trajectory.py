import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams.update({'font.size': 20,
                         'font.weight': 'normal',
                         'font.family': 'sans-serif'})
mpl.rcParams['pdf.fonttype'] = 42  # to make text editable in pdf output
#mpl.rcParams['font.sans-serif'] = ['Arial']  # to make it Arial

def plot_trajectory(ax,episode=0):
    #df = pd.read_csv(f'../../../eval_test_{episode}.csv', index_col=[0])
    df = pd.read_csv(f'../../../manual_AT_treatment_2norm_0', index_col=[0])
    x = np.arange(0,len(df))
    ax.fill_between(x, 0, df['Treatment'], color='orange', label='drug')
    ax.plot(x, (df['Type 0'] + df['Type 1']), 'k', label='total', linewidth=2)
    ax.plot(x, df['Type 0'], 'b', label='wt', linewidth=2)
    ax.plot(x, df['Type 1'], 'g', label='mut', linewidth=2)
    ax.set_xlabel('time')
    ax.set_ylabel('# Cells')


fig, ax = plt.subplots(figsize=(8, 4))
plot_trajectory(ax,0)

#fig, ax = plt.subplots(2,3,figsize=(8, 4))
#for i in range(6):
#    plot_trajectory(ax[i//3,i%3],i)

#fig.savefig(r'..\results\images\manual_AT.pdf',transparent=True)
plt.show()