import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
import matplotlib 

fig, ax = plt.subplots()
ax.set_prop_cycle(cycler('color', plt.get_cmap('viridis', 10).colors))
for k in range(7,13):
    filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\2022_growth_layer\20220110_growth_layer\10_01_2022_growth_layer_d'+str(k)+r'\output\all_data.h5'
    
        
    f = h5py.File(filename, 'r')

    #print(list(f['data'].keys()))

    dat = f['data/cells'];
    cells_look_info = pd.read_hdf(filename,'data/cells/cell1');
    
    read_radii = pd.read_hdf(filename,'data/radius')
    #print(read_radii)
    fronts = pd.read_hdf(filename,'data/front_cell_number')
    #print(fronts)
    read_growth_layer = pd.read_hdf(filename,'data/growth_layer');
    #print(read_growth_layer);
    Rad = np.array(read_radii)
    Growth = np.array(read_growth_layer); 
    num_cells = np.array(fronts); 
    ax.plot(Rad,Growth,label='D = '+str(k))
ax.legend()

h5File = 'growth_layer.h5'
f = h5py.File(h5File, 'r')
fig, ax = plt.subplots()
ax.set_prop_cycle(cycler('color', plt.get_cmap('viridis', 10).colors))
for key in f['data/growth']:
    df = pd.read_hdf(h5File, 'data/growth/' + key + '/red')
    print(f['data/growth/'+key].attrs['filename'])
    df2 = pd.read_hdf(h5File, 'data/growth/' + key + '/blue')
    tot = df.add(df2, axis=1, fill_value=0)

    r_w = df.mean(axis=1)
    b_w = df2.mean(axis=1)

    ax.plot(r_w)
    #ax.plot(b_w,'c')
plt.show()
