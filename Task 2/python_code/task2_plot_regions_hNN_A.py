#! /usr/bin/env python
#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from task2_hNN_A import task2_hNN_A

# Generate coordinates with limits
# Based on edges of Polygon B
x_values = np.arange(2.5,4,0.005)
y_values = np.arange(2.5,4,0.005)
xx, yy = np.meshgrid(x_values, y_values)
xx_flat = xx.flatten()
yy_flat = yy.flatten()
grid = np.array([[xx_flat[i], yy_flat[i]] for i in range(len(xx_flat))])


# Classify the points in grid
data = task2_hNN_A(grid)
data = data.reshape((x_values.shape[0], y_values.shape[0]))

fig = plt.figure()
plt.title('Decision Regions hNN A')
plt.xticks(np.arange(2.5, 4, 0.25), fontsize=8)
plt.yticks(np.arange(2.5, 4, 0.25),fontsize=8)


# Plot and safe image
c = plt.contourf(x_values, y_values, data, cmap=cm.YlGnBu )
proxy = np.array([plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in c.collections])
plt.legend(proxy[[0,-1]], ['Class 0', 'Class 1'])

plt.show()
plt.draw()
fig.savefig('t2_regions_hNN_A.pdf')
