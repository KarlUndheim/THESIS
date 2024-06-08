tree_species_1_1 = 'Training_Data_Ref/split_data/train/1/A_chan1_25689.txt'
tree_species_1_2 = 'Training_Data_Ref/split_data/train/1/A_chan2_25689.txt'
tree_species_1_3 = 'Training_Data_Ref/split_data/train/1/A_chan3_25689.txt'


import numpy as np
import matplotlib.pyplot as plt


full_tree = 'Training_Data_Ref/split_data/mrg_train/1/mrg_A_25689.txt'
""" full_tree = tree_species_1_1 """
points = np.loadtxt(full_tree, delimiter=',')
x,y,z = points[:, 0], points[:, 1], points[:, 2]

# minimum and maximum bounds
min_bounds = points.min(axis=0)
max_bounds = points.max(axis=0)

# Calculate the maximum range and use it to set uniform axes limits
max_range = np.array([max_bounds[i] - min_bounds[i] for i in range(3)]).max()
center = min_bounds + (max_bounds - min_bounds) / 2

# Plotting below:
fig = plt.figure(figsize=(10, 8), facecolor='white')
ax = fig.add_subplot(111, projection='3d', facecolor='white')

# Remove gridlines, 
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Set background color of the pane and axis to white
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')

# Remove color from the axes
ax.w_xaxis.line.set_visible(False)
ax.w_yaxis.line.set_visible(False)
ax.w_zaxis.line.set_visible(False)

# Plot points
ax.scatter(x, y, z, c='g', s=3)
ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)

ax.view_init(elev=0, azim=240)
ax.dist = 12 

plt.show()