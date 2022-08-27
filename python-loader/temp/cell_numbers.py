from load.pyMCDS import pyMCDS
import numpy as np
import matplotlib.pyplot as plt
import glob 
import pathlib
from plot import density
from time import time
from scipy.spatial import ConvexHull, convex_hull_plot_2d, distance
from scipy import interpolate

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]

# define the path
directory = pathlib.Path('data\output')

# choose a file to process
mcds1 = pyMCDS('output00000040.xml', directory)
"""
# commands to extract basic information
print(mcds1.get_time())

# commnds to extract pandas compatible output
print(mcds1.get_cell_df())
print(mcds1.get_cell_df_at(x=39, y=83, z=0))

# commands to extract mesh related output
print(mcds1.get_mesh_spacing())
print(mcds1.get_mesh())
print(mcds1.get_2D_mesh())
print(mcds1.get_linear_voxels())
print(mcds1.get_containing_voxel_ijk(x=0,y=0,z=0))
"""


xx, yy = mcds1.get_2D_mesh(); 
x_mesh_vals = xx[0]; 
y_mesh_vals = yy[:,0]; 

#print(x_mesh_vals)
#print(y_mesh_vals) 

cells = mcds1.get_cell_df();
x_s = np.array(cells['position_x'])
y_s = np.array(cells['position_y'])

rho, theta = cart2pol(x_s,y_s)

#sort them radially
idx = np.argsort(rho)
rho = rho[idx]
theta = theta[idx]
xs = x_s[idx]
ys = y_s[idx]
Colony_radius = rho[-1]
cell_diameter = 14
#plt.figure()
#plt.scatter(xs[rho > rho[-1]-300],ys[ rho > rho[-1]-300],marker='o', c='r');


points = np.concatenate([x_s,y_s]).reshape((2,len(x_s))).T;
hull = ConvexHull(points);
#_ = convex_hull_plot_2d(hull)

indices = np.unique(hull.simplices.flat)
hull_pts = points[indices,:]

hull_x = hull_pts[:,0];
hull_y = hull_pts[:,1]; 

hull_rho, hull_theta = cart2pol(hull_x,hull_y);

indx = np.argsort(hull_theta); 

hull_x_sorted= hull_x[indx]
hull_y_sorted= hull_y[indx]

#plt.figure()
#plt.plot(points[:, 0], points[:, 1], 'ko', markersize=2)
#plt.plot(hull_pts[:, 0], hull_pts[:, 1], 'ro', alpha=.25, markersize=20)

#interpolate
Number_of_points = int(2*np.pi*Colony_radius/cell_diameter)
#print(Number_of_points)
tck, u = interpolate.splprep([hull_x_sorted, hull_y_sorted], s=0, per=True);
xi, yi = interpolate.splev(np.linspace(0, 1, Number_of_points), tck)
fig, ax = plt.subplots(1, 1)
ax.plot(hull_pts[:, 0], hull_pts[:, 1], 'or')
ax.plot(xi, yi, '+b')

#nearest neighbors


Closest_points = np.zeros((Number_of_points,2))
cell_type_array = np.zeros(Number_of_points)
k = 0;
for xxx, yyy  in zip(xi,yi):
    pt = (xxx,yyy)
    Closest_points[k,0],Closest_points[k,1] = closest_node(pt, points);
    find_a_cell = cells[cells['position_x'] == Closest_points[k,0]];
    if len(find_a_cell)>1:
        find_a_cell = find_a_cell[find_a_cell['position_y'] == Closest_points[k,1]];
    cell_type_array[k] = find_a_cell['cell_type'];
   
    k+=1;
  

Closest_points, ind=np.unique(Closest_points, axis=0, return_index = True)

cell_type_array = cell_type_array[ind]


#print(Closest_points[0,:])
#print(cells['position_x']==Closest_points[0,0])
#print(cells['position_x']==Closest_points[0,0] and cells['position_y']==Closest_points[0,1])

plt.figure()
plt.plot(points[:, 0], points[:, 1], 'ko', markersize=2)
plt.plot(Closest_points[:, 0], Closest_points[:, 1], 'ro', alpha=.25, markersize=10)
plt.plot(xi,yi,'g+',markersize=4)

# reiterate. 
rho2, theta2 = cart2pol(Closest_points[:, 0], Closest_points[:, 1]);

indx = np.argsort(theta2); 

x_sorted= Closest_points[indx,0]
y_sorted= Closest_points[indx,1]

tck2, u2 = interpolate.splprep([x_sorted, y_sorted], s=0, per=True);
xi, yi = interpolate.splev(np.linspace(0, 1, Number_of_points), tck2)

#nearest neighbors


Closest_points = np.zeros((Number_of_points,2))
cell_type_array = np.zeros(Number_of_points)
k = 0;
for xxx, yyy  in zip(xi,yi):
    pt = (xxx,yyy)
    Closest_points[k,0],Closest_points[k,1] = closest_node(pt, points);
    find_a_cell = cells[cells['position_x'] == Closest_points[k,0]];
    if len(find_a_cell)>1:
        find_a_cell = find_a_cell[find_a_cell['position_y'] == Closest_points[k,1]];
    cell_type_array[k] = find_a_cell['cell_type'];
   
    k+=1;
  

Closest_points, ind=np.unique(Closest_points, axis=0, return_index = True)

cell_type_array = cell_type_array[ind]


#print(Closest_points[0,:])
#print(cells['position_x']==Closest_points[0,0])
#print(cells['position_x']==Closest_points[0,0] and cells['position_y']==Closest_points[0,1])

plt.figure()
plt.plot(points[:, 0], points[:, 1], 'ko', markersize=2)
plt.plot(Closest_points[:, 0], Closest_points[:, 1], 'ro', alpha=.25, markersize=10)
plt.plot(xi,yi,'g+',markersize=4)


#density.get_single_density_plot(cells, plt.cm.Reds) 

plt.show()
