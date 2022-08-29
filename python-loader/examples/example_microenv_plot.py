from load.pyMCDS import pyMCDS
from load.pyMCDS_timeseries import pyMCDS_timeseries
import numpy as np
import matplotlib.pyplot as plt
 
## load data


## Set our z plane and get our substrate values along it
def plot_microenv(): 

    z_val = 0.00
    chemical = 'drug'; 

    plane_drug = mcds.get_concentrations('drug', z_slice=z_val)

    ## Get the 2D mesh for contour plotting
    xx, yy = mcds.get_2D_mesh()

    # We want to be able to control the number of contour levels so we
    # need to do a little set up
    num_levels = 30

    
    # set up the figure area and add data layers
    fig, ax = plt.subplots()
    if chemical == 'drug': 
        min_conc = plane_drug.min()
        print(min_conc)
        max_conc = plane_drug.max()
        my_levels = np.linspace(min_conc, max_conc, num_levels)
        cs = ax.contourf(xx, yy, plane_drug, levels=my_levels)
        ax.contour(xx, yy, plane_drug, color='black', levels = my_levels,linewidths=0.5)
        ax.set_title('drug (mmHg) at t = {:.1f} {:s}, z = {:.2f} {:s}'.format(mcds.get_time(),mcds.data['metadata']['time_units'],z_val,mcds.data['metadata']['spatial_units']) )
    elif chemical == 'oxygen': 
        min_conc = plane_oxy.min()
        max_conc = plane_oxy.max()
        my_levels = np.linspace(min_conc, max_conc, num_levels)
        cs = ax.contourf(xx, yy, plane_oxy, levels=my_levels)
        ax.contour(xx, yy, plane_oxy, color='black', levels = my_levels,linewidths=0.5)
        ax.set_title('oxygen (mmHg) at t = {:.1f} {:s}, z = {:.2f} {:s}'.format(mcds.get_time(),mcds.data['metadata']['time_units'],z_val,mcds.data['metadata']['spatial_units']) )
        
    # Now we need to add our color bar
    cbar1 = fig.colorbar(cs, shrink=0.75)
    cbar1.set_label('mmHg')
    
    # Let's put the time in to make these look nice
    ax.set_aspect('equal')
    ax.set_xlabel('x (micron)')
    ax.set_ylabel('y (micron)')


mcds = pyMCDS('output00000012.xml', '.\..\..\PhysiCell_V_1.10.4\output')
plot_microenv();


 
plt.show()

