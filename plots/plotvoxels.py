tree_species_1_1 = 'Training_Data_Ref/split_data/train/1/B_chan1_3827.txt'
tree_species_1_2 = 'Training_Data_Ref/split_data/train/1/B_chan2_3827.txt'
tree_species_1_3 = 'Training_Data_Ref/split_data/train/1/B_chan3_3827.txt'


import numpy as np
import matplotlib.pyplot as plt

full_tree = 'Training_Data_Ref/split_data/mrg_val/1/mrg_A_4404.txt'
points = np.loadtxt(full_tree, delimiter=',')

# Parameters
voxel_size = 1.0
min_bounds = points.min(axis=0) - voxel_size
max_bounds = points.max(axis=0) + voxel_size

# Calculate the maximum range and use it to set uniform axes limits
max_range = np.array([max_bounds[i] - min_bounds[i] for i in range(3)]).max()
center = min_bounds + (max_bounds - min_bounds) / 2

# Create a voxel grid
grid_dimensions = ((max_bounds - min_bounds) / voxel_size).astype(int)
voxel_grid = np.zeros(grid_dimensions, dtype=bool)

# Fill the voxel grid
for point in points:
    indices = np.floor((point - min_bounds) / voxel_size).astype(int)
    if (indices >= 0).all() and (indices < grid_dimensions).all(): 
        voxel_grid[tuple(indices)] = True

# Plotting below:
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Add voxels
x, y, z = np.indices(np.array(voxel_grid.shape) + 1) * voxel_size + min_bounds[:, None, None, None]
ax.voxels(x, y, z, voxel_grid, facecolors='g', edgecolor='g', alpha=0.2)

# Plot points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='k', s=3) 
ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)
ax.view_init(elev=20, azim=240)
ax.dist = 5  
ax.set_xlabel('X Coordinates')
ax.set_ylabel('Y Coordinates')
ax.set_zlabel('Z Coordinates')
plt.show()