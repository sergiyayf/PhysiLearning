import matplotlib as mpl
from networkx.algorithms.structuralholes import constraint

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# fig, ax = plt.subplots()
# df = pd.read_hdf('./Evaluations/KppEnvEval__mtd.h5', key='run_0')
# time = np.arange(0, len(df['Type 0']))/2000
#
# ax.plot(time, df['Type 0'], c='blue')
# ax.plot(time, df['Type 1'], c='orange')
# ax.set_yscale('linear')
runs = 5
fig, ax = plt.subplots(runs, 1, constrained_layout=True)
for i in range(runs):
    df = pd.read_hdf('./Evaluations/KppEnvEval_2025_03_19_lin_5.h5', key='run_'+str(i))
    day = 400
    time = np.arange(0, len(df['Type 0']))/day
    time = time[::day]
    sens = np.array(df['Type 0'][::day])
    res = np.array(df['Type 1'][::day])
    tot = sens+res
    tot = tot
    treat = np.array(df['Treatment'][::day])

    if runs == 1:
        a = ax
    else:
        a = ax[i]
    a.plot(time, sens, c='blue')
    a.plot(time, res, c='orange')
    a.plot(time, tot, '--', c='k')
    a.set_yscale('linear')
    a.axhline(y=3.0, color='k')
    a.axhline(y=1.0, color='k')
    a.set_ylim(0, 4)
    #a.set_xlim(0, 60)
    #fill_between treatment
    for t in range(len(time)-1):
        if treat[t] == 1:
            a.axvspan((t-1), t, color='grey', alpha=0.5)
    #ax[i].fill_between(time, 0, 3, where=treat==1, color='gray', alpha=0.5)
# read npy
# dens = np.load('./Evaluations/KppEnvEval_ttp_8/density_4.npy')
# s_0 = dens[:,0,0]
# r_0 = dens[:,0,1]
# s_1 = dens[:,43000+1,0]
# r_1 = dens[:,43000+1,1]
# fig, ax = plt.subplots(2,2)
# ax[0,0].plot(s_0, c='blue')
# ax[0,0].plot(r_0, c='orange')
# ax[0,0].set_yscale('linear')
# ax[0,1].plot(s_1, c='blue')
# ax[0,1].plot(r_1, c='orange')
# ax[0,1].set_yscale('linear')
fig.savefig('agent_lin_1903a5.pdf', transparent=True)
plt.show()