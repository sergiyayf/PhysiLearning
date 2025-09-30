import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    df = pd.read_hdf('./Evaluations/LvEnvEval__lv_mtd.h5', key=f'run_{i}')
    skip = 800 # do every 400 frames, which represent 1 day
    sus = df['Type 0'].values[::skip]
    res = df['Type 1'].values[::skip]
    day = np.arange(1,len(sus)+1)
    time = day
    tot = sus + res
    #fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax[i].plot(day, sus, label='Wild Type', color='#5E82B8')
    ax[i].plot(day, res, label='Resistant Type', color='#EBAA42')
    ax[i].plot(day, tot, label='Total', color='k')
    # fill between treat
    treat = np.array(df['Treatment'].values[::skip])
    treat = treat[:len(time)]
    # replace 0s that are directly after 1 with 1s
    #treat = np.where(treat == 0, np.roll(treat, -1), treat)
    for t in range(len(time)-1):
        if treat[t] == 1:
            ax[i].axvspan((t-1), t, color='#A8DADC')
    ax[i].set_xlabel('Days')
    ax[i].set_ylabel('Population')
    ax[i].set_xlim(0, 100)
plt.show()