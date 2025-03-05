import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    # Load data
    os.chdir('/')
    df = pd.read_pickle('for_velocity_plotting_10_nc_runs_df.pkl')
    # Define grid size
    grid_size = 10

    # Create grid
    #x = np.arange(df['position_x'].min(), df['position_x'].max(), grid_size)
    #y = np.arange(df['position_y'].min(), df['position_y'].max(), grid_size)
    x = np.arange(-600, 650, grid_size)
    y = np.arange(-600, 650, grid_size)
    grid_x, grid_y = np.meshgrid(x, y)

    # Map cell positions to grid
    df['grid_x'] = pd.cut(df['position_x'], bins=x, labels=False, include_lowest=True)
    df['grid_y'] = pd.cut(df['position_y'], bins=y, labels=False, include_lowest=True)

    # Group by grid cell and calculate average velocity
    grouped = df.groupby(['grid_x', 'grid_y']).agg({'velocity_x': 'mean', 'velocity_y': 'mean'}).reset_index()
    # quiver plot of the average velocity, coloring the arrows with viridis, handling outliers
    # Calculate the magnitude of the velocity vectors
    magnitude = np.sqrt(grouped['velocity_x']**2 + grouped['velocity_y']**2)

    # Determine a reasonable upper limit for the color scale
    upper_limit = np.percentile(magnitude, 99)

    # Create a colormap reversed viridis
    cmap = plt.cm.Blues

    # Normalize the color scale to the upper limit
    norm = mpl.colors.Normalize(vmin=0, vmax=upper_limit)

    # Create a colorbar with the normalized data
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot the quiver plot with the normalized color scale
    ax.quiver(grouped['grid_x'], grouped['grid_y'],
               grouped['velocity_x'], grouped['velocity_y'], magnitude, cmap=cmap, norm=norm)
    # put correct coordinate ticks
    ax.set_xticks(np.arange(0, len(x), 20))
    ax.set_xticklabels(x[::20])
    ax.set_yticks(np.arange(0, len(y), 20))
    ax.set_yticklabels(y[::20])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # plot the colorbar
    fig.colorbar(sm)
    # equal aspect ratio
    ax.axis('equal')