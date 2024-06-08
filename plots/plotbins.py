import numpy as np
import matplotlib.pyplot as plt

# Load your data
full_tree = 'Training_Data_Ref/split_data/mrg_train/1/mrg_B_8659.txt'
points = np.loadtxt(full_tree, delimiter=',')
x,y,z = points[:, 0], points[:, 1], points[:, 2]

# minimum and maximum bounds
min_bounds = points.min(axis=0)
max_bounds = points.max(axis=0)

# Set number of bins
num_bins_z = 10 

# Calculate the maximum range to set uniform axes limits
max_range = np.array([max_bounds[i] - min_bounds[i] for i in range(3)]).max()
center = min_bounds + (max_bounds - min_bounds) / 2


# Plotting below:
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each bin as a separate segment
bin_height = max_range / num_bins_z
for i in range(num_bins_z):
    z_start = center[2] - max_range / 2 + i * bin_height
    ax.bar3d(center[0] - max_range / 2, center[1] - max_range / 2, z_start, 
             max_range, max_range, bin_height, edgecolor='g', color='gray' ,alpha=0.1)

# Plot points
ax.scatter(x, y, z, c='k', s=3) 

# Set uniform axes limits
ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)
ax.set_box_aspect([1,1,1]) 

# Adjust view and distance
ax.view_init(elev=10, azim=240)
ax.dist = 10 

ax.set_xlabel('X Coordinates')
ax.set_ylabel('Y Coordinates')
ax.set_zlabel('Z Coordinates')
plt.show()
