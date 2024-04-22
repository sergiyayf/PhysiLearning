import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import os
ex = {'font.size': 6,
          'font.weight': 'normal',
          'font.family': 'sans-serif'}
plt.rcParams.update(ex)
mpl.rcParams['pdf.fonttype'] = 42  # to make text editable in pdf output
mpl.rcParams['font.sans-serif'] = ['Arial']  # to make it Arial
os.chdir('/Users/saif/Desktop/Serhii/Projects/PhysiLearning')

# read full pcdl data
run = 1
with h5py.File(f'./data/position_physilearning/run_{run}/Evaluations/sim_full_data/pcdl_data_job_7676594_port_0.h5', 'r') as f:
    runs = list(f.keys())
    times = list(f['run_4'].keys())

timer = 0
times = times[:-45]
color = [plt.get_cmap('YlOrRd')(x) for x in np.linspace(0.1, 1, len(times))]
# sort times
df_radial_positions = pd.DataFrame()
times = sorted(times, key=lambda x: int(x.split('_')[-1]))
for time in times:
    ddf = pd.read_hdf(f'./data/position_physilearning/run_{run}/Evaluations/sim_full_data/pcdl_data_job_7676594_port_0.h5', f'run_4/{time}')
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
fig, ax = plt.subplots(figsize=(500 / 72, 150 / 72), constrained_layout=True)
sns.kdeplot(data=df_radial_positions, x='radial_position', hue='time', common_norm=True, ax=ax, palette=color, linewidth=1.5)
plt.xlabel('Radial position')
plt.title('KDE of mutants radial positions')
ax.legend([])
fig.savefig(r'fig2_kde_plot_mutant_position_evolution.pdf', transparent = True)
plt.show()

