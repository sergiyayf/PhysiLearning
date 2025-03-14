import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig, ax = plt.subplots()
df = pd.read_hdf('./Evaluations/KppEnvEval__mtd.h5', key='run_0')
# ax.plot(df['Type 0'])
ax.plot(df['Type 1'], c='b')

df = pd.read_hdf('./Evaluations/KppEnvEval__at_50.h5', key='run_0')
# ax.plot(df['Type 0'])
ax.plot(df['Type 1'], c='r')

df = pd.read_hdf('./Evaluations/KppEnvEval__eat100.h5', key='run_0')
# ax.plot(df['Type 0'])
ax.plot(df['Type 1'], c='g')

df = pd.read_hdf('./Evaluations/KppEnvEval__at100.h5', key='run_0')
# ax.plot(df['Type 0'])
ax.plot(df['Type 1'], c='k')

df = pd.read_hdf('./Evaluations/KppEnvEval_atagnt2025_kpp_test_training.h5', key='run_1')
ax.plot(df['Type 0']/100, c='c')
ax.plot(df['Type 1']/100, c='m')
ax.set_yscale('log')

# read npy trajectory
trajectory = np.load('./Evaluations/KppEnvEval__mtd_density.npy')

plt.show()