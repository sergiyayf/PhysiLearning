"""   Note that sufficiently large cell numbers are needed to efficiently compute the periferies """


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

def FrontCells(cells):
    """ this function takes cells dataframe as an input; 
    it searches for the cells on the perifery by first creating a concave boundary of the colony, then interpolating points inbetween and at the end reiterating the procedure, for the new found points to fill the gaps between the cells 
    it returns a (N,2) array, N is initially not known, it depends on the colony size of cells position, 
    and another (N,1) array of cell types. 
    """
    x_s = np.array(cells['position_x'])
    y_s = np.array(cells['position_y'])

    rho, theta = cart2pol(x_s,y_s)

    #sort them radially
    idx = np.argsort(rho)
    rho = rho[idx]
    theta = theta[idx]
    xs = x_s[idx]
    ys = y_s[idx]

    # Roughly calculate colony radius to estimate number of cells
    Colony_radius = rho[-1]
    cell_diameter = 14

    #get a concave boundary as a null boundary to further iterate on
    points = np.concatenate([x_s,y_s]).reshape((2,len(x_s))).T;
    hull = ConvexHull(points);
    #_ = convex_hull_plot_2d(hull)

    # sort those points to be able to interpolate
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

    #interpolate points of concave boundary with splines, use estimated number of cells on the perifery to create N points of the interpolated curve
    Number_of_points = int(2*np.pi*Colony_radius/cell_diameter)
    #print(Number_of_points)
    tck, u = interpolate.splprep([hull_x_sorted, hull_y_sorted], s=0, per=True);
    xi, yi = interpolate.splev(np.linspace(0, 1, Number_of_points), tck)
    #fig, ax = plt.subplots(1, 1)
    #ax.plot(hull_pts[:, 0], hull_pts[:, 1], 'or')
    #ax.plot(xi, yi, '+b')

    # calculate nearest neighbors to the interpolated points 

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

    #plt.figure()
    #plt.plot(points[:, 0], points[:, 1], 'ko', markersize=2)
    #plt.plot(Closest_points[:, 0], Closest_points[:, 1], 'ro', alpha=.25, markersize=10)
    #plt.plot(xi,yi,'g+',markersize=4)

    # reiterate on that to get more cells from the boundary and dont have holes. so next is basicaly the same code
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
    

    # this is the final result of (N,2) array Closest_points - for x and y positions of the cells at the boundary. and cell_type_array for (N,1) for the types of those cells. If needed other properties could be found then by finding full dataframes that correspond to that particular cell position. 

    Closest_points, ind=np.unique(Closest_points, axis=0, return_index = True)
    cell_type_array = cell_type_array[ind]
    
    return Closest_points, cell_type_array; 

def VisualizeFront(cells):
    """ this function takes cell dataframe as an input and shows the cells, that are thought to be the front ones """
    clls = cells;
    x_pos = clls['position_x'];
    y_pos = clls['position_y'];
    periferial_cells, cell_types = FrontCells(clls);   
    plt.figure()
    plt.plot(x_pos, y_pos, 'ko', markersize=2)
    plt.plot(periferial_cells[:, 0], periferial_cells[:, 1], 'ro', alpha=.25, markersize=10)

