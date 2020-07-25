#! /usr/bin/env python
#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from task2_sNN_AB import task2_sNN_AB


# Generate points between 0 and 10 to classify
x_values = np.arange(-2,8, 0.005)
y_values = np.arange(-3,7,0.005)
xx, yy = np.meshgrid(x_values, y_values)
xx = xx.flatten()
yy = yy.flatten()
grid = np.array([[xx[i], yy[i]] for i in range(len(xx))])


# Classify the points and reshape the result to fit the plot function.
data = task2_sNN_AB(grid)
data = data.reshape((x_values.shape[0], y_values.shape[0]))

# Setup the plot title and axis
fig, ax = plt.subplots()
plt.title('Decision Regions sNN AB')
plt.xticks(np.arange(-2, 8, 1), fontsize=8)
plt.yticks(np.arange(-3, 7, 1),fontsize=8)


# Plot data and show result
c = plt.contourf(x_values, y_values, data, cmap=cm.YlGnBu )
proxy = np.array([plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in c.collections])
plt.legend(proxy[[0,-1]], ['Class 0', 'Class 1'])

   # PLOT USED FOR DEBUGGING
# f = open('task2_data.txt')
# lines = []
# for line in f:
#     lines.append(line.split())
# f.close()
# PolygonA = np.zeros((int((len(lines[0])-1)/2),2))
# PolygonB = np.zeros((int((len(lines[1])-1)/2),2))
# PolygonA[:,0] = np.array([float(lines[0][i]) for i in range(1,len(lines[0])) if i%2 == 1])
# PolygonA[:,1] = np.array([float(lines[0][i]) for i in range(1,len(lines[0])) if i%2 == 0])
# PolygonB[:,0] = np.array([float(lines[1][i]) for i in range(1,len(lines[1])) if i%2 == 1])
# PolygonB[:,1] = np.array([float(lines[1][i]) for i in range(1,len(lines[1])) if i%2 == 0])


# ax.plot(PolygonA[[0,1],0], PolygonA[[0,1],1], c = 'red')
# ax.plot(PolygonA[[1,2],0], PolygonA[[1,2],1], c = 'red')
# ax.plot(PolygonA[[2,3],0], PolygonA[[2,3],1], c = 'red')
# ax.plot(PolygonA[[0,3],0], PolygonA[[0,3],1], c = 'red')
# ax.plot(PolygonB[[0,1],0], PolygonB[[0,1],1], c = 'green')
# ax.plot(PolygonB[[1,2],0], PolygonB[[1,2],1], c = 'green')
# ax.plot(PolygonB[[2,3],0], PolygonB[[2,3],1], c = 'green')
# ax.plot(PolygonB[[0,3],0], PolygonB[[0,3],1], c = 'green')

plt.show()
plt.draw()
fig.savefig('t2_regions_sNN_AB.pdf')

