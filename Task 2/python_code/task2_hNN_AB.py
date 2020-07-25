#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io
from task2_hNeuron import task2_hNeuron
import matplotlib.pyplot as plt
import random

def task2_hNN_AB(X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    # Hard coding precalculated weights for Layer 1
    weightAL1Matrix = np.array([[ 1.,-0.68871342, -4.63282829, 1.], [-0.17355183, 1., 0.52779144,-0.49109806], [-0.10824754, -0.75116169, 1.,0.2298064 ]])
    weightBL1Matrix = np.array([[ 2.40503227, -0.99634817, -2.69037292, 7.70803051], [-1., 1., 1., -0.69669645],[ 0.98876441, -0.38886998, 0.18348208,-1.]])

    weightL1Matrix = np.ones((weightAL1Matrix.shape[0], weightAL1Matrix.shape[1] + weightBL1Matrix.shape[1]))
    weightL1Matrix[:, [0,1,2,3]] = weightAL1Matrix[:]
    weightL1Matrix[:, [4,5,6,7]] = weightBL1Matrix[:]
    
    weightL2Matrix = np.ones((9,2))
    weightL2Matrix[:,0] = np.array([-3.9,1,1,1,1,0,0,0,0])
    weightL2Matrix[:,1] = np.array([-4.9, 0, 0, 0, 0, 2, 1, 1, 2])

    weightL3 = np.array([-0.5,-1,1]).reshape((3,1))

    # Calculating output of First hidden layer
    L1 = np.zeros((X.shape[0], weightL1Matrix.shape[1]))

    for i in range(L1.shape[1]):
        L1[:,i] = task2_hNeuron(weightL1Matrix[:,i], X).reshape((X.shape[0],))
    
    # Calculating output of Second hidden layer
    L2 = np.zeros((X.shape[0], weightL2Matrix.shape[1]))

    for i in range(L2.shape[1]):
        L2[:,i] = task2_hNeuron(weightL2Matrix[:,i], L1).reshape((L2.shape[0],))

    # Calculating Final output
    Y = task2_hNeuron(weightL3, L2)

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
    # print(Y)
    # fig, ax = plt.subplots()
    # ax.plot(PolygonA[[0,1],0], PolygonA[[0,1],1], c = 'red')
    # ax.plot(PolygonA[[1,2],0], PolygonA[[1,2],1], c = 'red')
    # ax.plot(PolygonA[[2,3],0], PolygonA[[2,3],1], c = 'red')
    # ax.plot(PolygonA[[0,3],0], PolygonA[[0,3],1], c = 'red')
    # ax.plot(PolygonB[[0,1],0], PolygonB[[0,1],1], c = 'blue')
    # ax.plot(PolygonB[[1,2],0], PolygonB[[1,2],1], c = 'blue')
    # ax.plot(PolygonB[[2,3],0], PolygonB[[2,3],1], c = 'blue')
    # ax.plot(PolygonB[[0,3],0], PolygonB[[0,3],1], c = 'blue')

    # plt.scatter(X[:,0], X[:,1], c='red')
    # for i in range(X.shape[0]):
    #     ax.annotate( i, (X[i][0], X[i][1]))
    # plt.show()

    return Y
