#! /usr/bin/env python
#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import task2_hNeuron

def find_weight(x1,x2):
    '''
    Finds the weights for the decision boundary of 
    Parameters:
    x1, x2 - 1x2 arrays of the coordinates of two points defining a line (the decision boundary)
    '''
    # parameters of the cartesian form of the boundary line
    # b = (x2[1] - x1[1]*x2[0]/x1[0])*x1[0]/(x1[0] - x2[0]) 
    # a = (x1[1]-b)/x1[0]
    a = (x1[1] - x2[1])/(x1[0] - x2[0])
    b = -a*x1[0] + x1[1] 
    w1 = 1
    w2 = -w1/a # assigning w2 so that [w1, w2] is ortogonal to the line vector
    w0 = -b*w2 # assigning the bias
    weight = np.array([w0, w1, w2])
    weight = weight/np.amax(weight)
    return weight

def full_weightsA(PolygonA):

    weight0 = find_weight(PolygonA[0,:], PolygonA[1,:])
    weight0 = -weight0
    weight0 = weight0/np.amax(weight0)
    weight1 = find_weight(PolygonA[1,:], PolygonA[2,:])
    weight2 = find_weight(PolygonA[2,:], PolygonA[3,:])
    weight3 = find_weight(PolygonA[3,:], PolygonA[0,:])
    weight3 = -weight3
    weight3 = weight3/np.amax(weight3)

    weightM = np.zeros((3, 4))
    weightM[:,0] = weight0
    weightM[:,1] = weight1
    weightM[:,2] = weight2
    weightM[:,3] = weight3

    weightL2 = np.array([-3.90, 1,1,1,1])
    return weightM, weightL2

def full_weightsB(PolygonB):
    weight0 = find_weight(PolygonB[0,:], PolygonB[1,:])
    weight0 = -weight0
    weight0 = weight0/np.amax(weight0)
    weight1 = find_weight(PolygonB[1,:], PolygonB[2,:])
    weight2 = find_weight(PolygonB[2,:], PolygonB[3,:])
    weight3 = find_weight(PolygonB[3,:], PolygonB[0,:])
    weight3 = -weight3
    weight3 = weight3/np.amax(weight3)

    weightM = np.zeros((3, 4))
    weightM[:,0] = weight0
    weightM[:,1] = weight1
    weightM[:,2] = weight2
    weightM[:,3] = weight3

    weightL2 = np.array([-4.90, 2, 1, 1, 2])
    print(weightM)

    return weightM, weightL2


def save_to_file(weightM, weightL2):
    for column in range(weightM.shape[1]):
        for row in range(weightM.shape[0]):
            if column == 0 and row == 0:
                f = open('task2 hNN A weights.txt', 'w')
            else:
                f = open('task2 hNN A weights.txt', 'a')
            f.write(f'W(1,{column+1},{row}) : {weightM[row][column]}\n')
            f.close()

    for idx,weight in enumerate(weightL2):
        f = open('task2 hNN A weights.txt', 'a')
        f.write(f'W(2,{1},{idx}) : {weight}\n')


# def main():

#     f = open('task2_data.txt')
#     lines = []
#     for line in f:
#         lines.append(line.split())
#     f.close()
#     PolygonA = np.zeros((int((len(lines[0])-1)/2),2))
#     PolygonB = np.zeros((int((len(lines[1])-1)/2),2))
#     PolygonA[:,0] = np.array([float(lines[0][i]) for i in range(1,len(lines[0])) if i%2 == 1])
#     PolygonA[:,1] = np.array([float(lines[0][i]) for i in range(1,len(lines[0])) if i%2 == 0])
#     PolygonB[:,0] = np.array([float(lines[1][i]) for i in range(1,len(lines[1])) if i%2 == 1])
#     PolygonB[:,1] = np.array([float(lines[1][i]) for i in range(1,len(lines[1])) if i%2 == 0])
#     weightAM, weightAL2 = full_weightsA(PolygonA)
#     weightBM, weightBL2 = full_weightsB(PolygonB)
#     fig, ax = plt.subplots()
#     ax.plot(PolygonA[[0,1],0], PolygonA[[0,1],1], c = 'red')
#     ax.plot(PolygonA[[1,2],0], PolygonA[[1,2],1], c = 'red')
#     ax.plot(PolygonA[[2,3],0], PolygonA[[2,3],1], c = 'red')
#     ax.plot(PolygonA[[0,3],0], PolygonA[[0,3],1], c = 'red')
#     ax.plot(PolygonB[[0,1],0], PolygonB[[0,1],1], c = 'blue')
#     ax.plot(PolygonB[[1,2],0], PolygonB[[1,2],1], c = 'blue')
#     ax.plot(PolygonB[[2,3],0], PolygonB[[2,3],1], c = 'blue')
#     ax.plot(PolygonB[[0,3],0], PolygonB[[0,3],1], c = 'blue')
#     plt.show()

#     X = np.array([1.58, 2.8]).reshape(1,2)
#     print(task2_hNeuron.task2_hNeuron(weightBM[:,0], X))
#     print(task2_hNeuron.task2_hNeuron(weightBM[:,1], X))
#     print(task2_hNeuron.task2_hNeuron(weightBM[:,2], X))
#     print(task2_hNeuron.task2_hNeuron(weightBM[:,3], X))


# main()




