import os

import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic_2d
ex = {'font.size': 6,
          'font.weight': 'normal',
          'pdf.fonttype': 42,
          'font.family': 'sans-serif',
          'font.sans-serif': 'Arial',
          'errorbar.capsize': 2,
          }
plt.rcParams.update(ex)
def get_ttps(filename, timesteps=50):
    ttps = []
    for i in range(timesteps):
        df = pd.read_hdf(filename, key=f'run_{i}')
        # find the largest index with non-zero Type 0 and Type 1
        initial_size = df['Type 0'][0] + df['Type 1'][0]
        nz = df[((df['Type 1']) / initial_size >= 1.0)]
        tot_idx = df[((df['Type 0'] + df['Type 1']) / initial_size >= 2.0)].index
        if len(nz) > 0:
            # append index when type 0 + type 1 is larger than 1.5
            ttps.append(nz.index[0] / 4)
        elif len(tot_idx) > 0:
            ttps.append(tot_idx[0] / 4)
        else:
            ttps.append(len(df) / 4)

    return ttps

os.chdir('/home/saif/Projects/PhysiLearning')
# set up color cycler with tab20b
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20b.colors[8:12])

num = 1
#fig, ax = plt.subplots(1, 1, figsize=(10,5))
#ax.set_title('Time to progression, vary position and maybe agents')
dft = pd.DataFrame()
# fig, ax = plt.subplots(10,20, figsize=(200, 100))
# get blues colors of length 5
colors = plt.cm.Blues(np.linspace(0.2, 1, 10))
markers = ['o' for i in range(10)]
ttps = []
resistant = []
susceptible = []
growth_rate = []
max_t = 400
fig, ax = plt.subplots(figsize=(120/72,120/72), constrained_layout=True)
for i in range(1,11):
    filename = f'./Evaluations/1402_lvs_evals/LvEnvEval__20250206_lv_1_{i}.h5'
    #filename = f'./Evaluations/1402_pcs_evals/run_{i}.h5'
    ttp = get_ttps(filename)
    ttps.append(ttp)
    #ax.scatter([i]*len(ttp), ttp, label=position, color=colors[j], marker=markers[i])
    #ax.scatter(i, np.mean(ttp), color=colors[j], marker='x')
    for k in range(50):
        df = pd.read_hdf(filename, key=f'run_{k}')
        tot = df['Type 0'] + df['Type 1']
        res = df['Type 1']
        sus = df['Type 0']
        sus = np.array(sus[tot > 0])
        res = np.array(res[tot > 0])
        tot = np.array(tot[tot > 0])
        # resistant growth
        res_growth = res[1:] / res[:-1] - 1
        ax.plot(range(len(res)), res)
        ax.set_yscale('log')
        # scatter plot total cell number vs resistant color the dots depending on the sine of res_growth - 1 binary
        blues = np.where(res_growth > 1)
        reds = np.where(res_growth < 1)
        #ax2.scatter(res[1:], res_growth)
        #ax2.scatter(sus[reds], res[reds], color='r', marker='o', s=1, alpha=0.5)

        for cell in range(len(res)-1):
            resistant.append(res[cell])
            susceptible.append(sus[cell])
            growth_rate.append(res_growth[cell])

# Define grid
y_bins = np.linspace(min(resistant), max(resistant), 100)  # X-axis bins
x_bins = np.linspace(min(susceptible), max(susceptible), 100)  # Y-axis bins
# Compute binned average of growth rates
stat, x_edges, y_edges, binnumber = binned_statistic_2d(
    susceptible, resistant, growth_rate, statistic='mean', bins=[x_bins, y_bins]
)
ax.set_xlabel('Time')
ax.set_ylabel('Resistant cells')
fig.savefig(f'./scripts/figures/plots/Figure_2_log_resistant_lv.pdf', transparent=True)
# Plot heatmap of binned growth rates
fig, ax = plt.subplots(figsize=(120/72, 120/72), constrained_layout=True)
# normalize colormap
vmax = np.nanmax(abs(stat))
vmin = -vmax
norm = plt.Normalize(vmin=vmin, vmax=vmax)

c = ax.pcolormesh(x_bins, y_bins, stat.T, cmap='RdBu', shading='auto', norm=norm)
# plot colormap
ax.set_xlabel('Sensitive cells')
ax.set_ylabel('Resistant cells')
# fig.colorbar(c, ax=ax, label='Growth rate')
fig.savefig(f'./scripts/figures/plots/Figure_2_heatmap_resistant_lv.pdf', transparent=True)

plt.show()
