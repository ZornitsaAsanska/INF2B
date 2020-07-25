#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io
import task2_hNeuron
import matplotlib.pyplot as plt

def task2_hNN_A(X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    # Hardcoding precalculated weights
    weightL1Matrix = np.array([[ 1.,-0.68871342, -4.63282829, 1.], [-0.17355183, 1., 0.52779144,-0.49109806], [-0.10824754, -0.75116169, 1.,0.2298064 ]])
    weightL2 = np.array([-3.9, 1, 1, 1, 1])

    # Calculating output of fisrt layer in L1
    # L1: N-by-4 matrix 
    L1 = np.zeros((X.shape[0], weightL1Matrix.shape[1]))

    for i in range(L1.shape[1]):
        L1[:,i] = task2_hNeuron.task2_hNeuron(weightL1Matrix[:,i], X).reshape((X.shape[0],))
    
    # Calculatig final output
    Y = task2_hNeuron.task2_hNeuron(weightL2, L1)

    # PLOT USED FOR DEBUGGING
    # f = open('task2_data.txt')
    # lines = []
    # for line in f:
    #     lines.append(line.split())
    # f.close()
    # PolygonA = np.zeros((int((len(lines[0])-1)/2),2))
    # PolygonA[:,0] = np.array([float(lines[0][i]) for i in range(1,len(lines[0])) if i%2 == 1])
    # PolygonA[:,1] = np.array([float(lines[0][i]) for i in range(1,len(lines[0])) if i%2 == 0])
    # print(Y)
    # print(X)
    # fig, ax = plt.subplots()
    # ax.plot(PolygonA[[0,1],0], PolygonA[[0,1],1], c = 'blue')
    # ax.plot(PolygonA[[1,2],0], PolygonA[[1,2],1], c = 'blue')
    # ax.plot(PolygonA[[2,3],0], PolygonA[[2,3],1], c = 'blue')
    # ax.plot(PolygonA[[0,3],0], PolygonA[[0,3],1], c = 'blue')

    # plt.scatter(X[:,0], X[:,1], c='red')
    # for i in range(X.shape[0]):
    #     ax.annotate( i, (X[i][0], X[i][1]))
    # plt.show()
    return Y

