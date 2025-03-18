import matplotlib as mpl
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

fig, ax = plt.subplots(10,1)
for i in range(10):
    df = pd.read_hdf('./Evaluations/KppEnvEval_ttp_1.h5', key='run_'+str(i))
    day = 400
    time = np.arange(0, len(df['Type 0']))/day
    time = time[::day]
    sens = np.array(df['Type 0'][::day])
    res = np.array(df['Type 1'][::day])
    tot = sens+res
    treat = np.array(df['Treatment'][::day])

    ax[i].plot(time, sens, c='blue')
    ax[i].plot(time, res, c='orange')
    ax[i].plot(time, tot, '--', c='k')
    ax[i].set_yscale('linear')
    ax[i].axhline(y=2.5, color='k')
    #fill_between treatment
    for t in range(len(time)-1):
        if treat[t] == 1:
            ax[i].axvspan((t-1), t, color='grey', alpha=0.5)
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

plt.show()