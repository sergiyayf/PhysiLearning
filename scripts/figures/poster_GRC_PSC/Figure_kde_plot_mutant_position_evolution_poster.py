import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import os
ex = {'font.size': 14,
          'font.weight': 'normal',
          'font.family': 'sans-serif'}
plt.rcParams.update(ex)
mpl.rcParams['pdf.fonttype'] = 42  # to make text editable in pdf output
mpl.rcParams['font.sans-serif'] = ['Arial']  # to make it Arial
os.chdir('/media/saif/1A6A95E932FFC943/09012025_onehalf_train_6_physicells/full_run_1/Evaluations/sim_full_data')

# read full pcdl data
run = 1
with h5py.File(f'./pcdl_data_job_14948503_port_0.h5', 'r') as f:
    runs = list(f.keys())
    times = list(f['run_3'].keys())

timer = 0
times = times[:-25]
color = [plt.get_cmap('YlOrRd')(x) for x in np.linspace(0.1, 1, len(times))]
# sort times
df_radial_positions = pd.DataFrame()
times = sorted(times, key=lambda x: int(x.split('_')[-1]))
for time in times:
    ddf = pd.read_hdf(f'./pcdl_data_job_14948503_port_0.h5', f'run_3/{time}')
    # store mutant positions
    mutant_x_positions = ddf[ddf['type'] == 1]['x'].values
    mutant_y_positions = ddf[ddf['type'] == 1]['y'].values
    radial_positions = np.sqrt(mutant_x_positions**2 + mutant_y_positions**2)

    # Add radial positions to the DataFrame
    df_radial_positions = pd.concat([df_radial_positions, pd.DataFrame({
        'time': float(time.split('_')[-1]) * np.ones(len(radial_positions)),
        'radial_position': radial_positions
    })], ignore_index=True)

# Plot KDE of radial positions for all times
fig, ax = plt.subplots(figsize=(8,2), constrained_layout=True)
sns.kdeplot(data=df_radial_positions, x='radial_position', hue='time', common_norm=True, ax=ax, palette=color, linewidth=2.0)
plt.xlabel('Radial position')
# create continuous colorbar
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=32))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
#plt.title('KDE of mutants radial positions')
ax.legend([])
# set format for y label to scientific notation
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
fig.savefig('/home/saif/Projects/PhysiLearning/plots/kde_plot_poster.pdf', transparent = True)
plt.show()

