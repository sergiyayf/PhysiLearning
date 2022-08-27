from load.pyMCDS import pyMCDS
import numpy as np
import matplotlib.pyplot as plt
import glob 
import pathlib
import os

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


# define the path
directory = pathlib.Path('data\switching_2')

# define the pattern
pattern = "output*.xml"

# list files with pattern in the directory
files = []
for output in directory.glob(pattern):
    
    a = os.path.basename(output)
    files.append(a)
    
# load data
mcdss = []
for eachfile in files:
    mc = pyMCDS(eachfile, directory)
    mcdss.append(mc)

# sort positions from the last output radially 

last_cell = mcdss[-1].get_cell_df()
z_val = 0.0
ds = mcdss[-1].get_mesh_spacing()

inside_plane = (last_cell['position_z'] < z_val + ds) & (last_cell['position_z'] > z_val - ds)
plane_cells = last_cell[inside_plane]

# We're going to plot two types of cells and we want it to look nice
colors = ['black', 'blue']
sizes = [20, 8]
labels = ['Alive', 'Dead']

# get our cells of interest
alive_cells = plane_cells[plane_cells['cycle_model'] <= 6]
dead_cells = plane_cells[plane_cells['cycle_model'] > 6]

plt.figure()
# plot the cell layer
for i, plot_cells in enumerate((alive_cells, dead_cells)):
    plt.scatter(plot_cells['position_x'].values, 
            plot_cells['position_y'].values, 
            facecolor='none', 
            edgecolors=colors[i],
            alpha=0.6,
            s=sizes[i],
            label=labels[i])


# get IDs and positions of cells
ID = np.array(plane_cells['ID'])
parent_ID = np.array(plane_cells['parent_ID']) 
x_s = np.array(plane_cells['position_x'])
y_s = np.array(plane_cells['position_y'])
zs = np.array(plane_cells['position_z']) 
cycle = np.array(plane_cells['cycle_model']) 

rho, theta = cart2pol(x_s,y_s)

#sort them radially
idx = np.argsort(rho)
rho = rho[idx]
theta = theta[idx]
xs = x_s[idx]
ys = y_s[idx]

#choose cells to track 
x_chosen_array = xs[len(xs)-300:len(xs)]
y_chosen_array = ys[len(ys)-300:len(ys)]

x_chosen = x_chosen_array
y_chosen = y_chosen_array 
    
plt.scatter( x_chosen, y_chosen, facecolors='red');

index1 = idx[len(idx)-300:len(idx)]
daughter_cell = index1;
tracking_cell = daughter_cell;
tracking_cell_mom = parent_ID[tracking_cell]
tracking_cell_mom = tracking_cell_mom.astype(int)
mom_cell = parent_ID[daughter_cell]
mom_cell = mom_cell.astype(int)
daughter_cell_x = x_s[daughter_cell];
daughter_cell_y = y_s[daughter_cell];
daughter_cell_x_old = x_s[daughter_cell];
daughter_cell_y_old = y_s[daughter_cell];

# now loop though all files 

for mcds in reversed(mcdss):
    #load cells, IDs and positions
    cells = mcds.get_cell_df()
    ID = np.array(cells['ID'])
    parent_ID = np.array(cells['parent_ID'])
    xx = np.array(cells['position_x'])
    yy = np.array(cells['position_y'])
    
    mom_cell_x = xx[mom_cell]
    mom_cell_y = yy[mom_cell]
    
    L = len(ID)
   
   
    for i in range(len(daughter_cell)):
        if daughter_cell[i]<L:
            daughter_cell_x[i] = xx[daughter_cell[i]]
            daughter_cell_y[i] = yy[daughter_cell[i]]
            plt.plot([daughter_cell_x[i], daughter_cell_x_old[i]], [daughter_cell_y[i], daughter_cell_y_old[i]])
            daughter_cell_x_old[i] = daughter_cell_x[i]
            daughter_cell_y_old[i] = daughter_cell_y[i]
        else:
            plt.plot([daughter_cell_x_old[i], mom_cell_x[i]], [daughter_cell_y_old[i], mom_cell_y[i]])
            daughter_cell[i] = mom_cell[i]
            mom_cell[i] = parent_ID[daughter_cell[i]]
            daughter_cell_x_old[i] = xx[daughter_cell[i]]
            daughter_cell_y_old[i] = yy[daughter_cell[i]]
            
   
print(index1)

#plt.plot([x_s[daughter_cell],x_s[mom_cell]], [y_s[daughter_cell], y_s[mom_cell]])
print(x_chosen_array)
print(len(ys))

plt.show()
