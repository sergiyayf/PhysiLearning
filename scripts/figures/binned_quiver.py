import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

# Example scattered data
resistant_cells = np.array([10, 20, 30, 40, 50, 60, 70, 30, 40, 20, 50, 60])
total_cells = np.array([50, 100, 150, 200, 250, 300, 350, 100, 200, 250, 150, 50])
growth_rate = np.array([0.8, 0.6, 0.5, 0.3, 0.2, 0.1, 0.05, 0.55, 0.25, 0.15, 0.4, 0.7])

# Define grid
x_bins = np.linspace(min(resistant_cells), max(resistant_cells), 10)  # X-axis bins
y_bins = np.linspace(min(total_cells), max(total_cells), 10)  # Y-axis bins

# Compute binned average of growth rates
stat, x_edges, y_edges, binnumber = binned_statistic_2d(
    resistant_cells, total_cells, growth_rate, statistic='mean', bins=[x_bins, y_bins]
)

# Grid center points
X, Y = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2)

# Create directional field (gradient-like effect)
U = np.gradient(stat, axis=1)  # Change in X-direction
V = np.gradient(stat, axis=0)  # Change in Y-direction

# Plot heatmap of binned growth rates
fig, ax = plt.subplots(figsize=(7, 6))
c = ax.pcolormesh(x_bins, y_bins, stat.T, cmap='viridis', shading='auto')

# Overlay quiver arrows for growth rate direction
ax.quiver(X, Y, U.T, V.T, color='white', scale=5)

# Add colorbar
cbar = plt.colorbar(c)
cbar.set_label('Average Growth Rate')

# Labels and title
ax.set_xlabel('Resistant Cells')
ax.set_ylabel('Total Cells')
ax.set_title('Binned Growth Rate with Quiver Directions')

plt.show()
