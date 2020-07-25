import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def task2_hNeuron(W, X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    #  W : (D+1)-by-1 vector of weights (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    Xext = np.ones((X.shape[0], X.shape[1] + 1))
    Xext[:,[i for i in range(1, X.shape[1]+1)]] = X
    step_parameter = Xext @ W
    step = np.vectorize(lambda x: 1 if x >0 else 0)
    Y = step(step_parameter)
    Y = Y.reshape(-1,1)
    return Y

def task2_sNeuron(W, X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    #  W : (D+1)-by-1 vector of weights (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    Xext = np.ones((X.shape[0], X.shape[1] + 1))
    Xext[:,[i for i in range(1, X.shape[1]+1)]] = X
    sigmoid_param = Xext @ W
    sigmoid_param = np.asarray(sigmoid_param, dtype=np.float128)
    sigmoid = np.vectorize(lambda x: 1/(1+np.exp(-x)))
    Y = sigmoid(sigmoid_param)
    Y = Y.reshape(-1,1)
    return Y

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
    '''
    Finds the weight vectors for the network of Polygon A
    Return:
    - weightM - 3-by-4 matrix of the weight vectors for each decision line as columns
    - weightL2 - weight vector hidden layer to output
    '''
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
    '''
    Finds the weight vectors for the network of Polygon B
    Return:
    - weightM - 3-by-4 matrix of the weight vectors for each decision line as columns
    - weightL2 - weight vector hidden layer to output
    '''
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
    '''
    Saves the weight matrix and final layer weight vector
    in .txt file given the specified format
    '''
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

def task2_hNN_A(X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    # Hardcoding precalculated weights
    weightL1Matrix = np.array([[ 1.,-0.68871342, -4.63282829, 1.], [-0.17355183, 1., 0.52779144,-0.49109806], [-0.10824754, -0.75116169, 1.,0.2298064 ]])
    weightL2 = np.array([-3.9, 1, 1, 1, 1])

    # Calculating output of first layer in L1
    # L1: N-by-4 matrix 
    L1 = np.zeros((X.shape[0], weightL1Matrix.shape[1]))

    for i in range(L1.shape[1]):
        L1[:,i] = task2_hNeuron(weightL1Matrix[:,i], X).reshape((X.shape[0],))
    
    # Calculatig final output
    Y = task2_hNeuron(weightL2, L1)

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

def task2_plot_regions_hNN_A():
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

def task2_plot_regions_hNN_AB():
    # Generate coordinates with limits
    # Based on edges of Polygon B
    x_values = np.arange(-2,8, 0.005)
    y_values = np.arange(-3,7,0.005)
    xx, yy = np.meshgrid(x_values, y_values)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    grid = np.array([[xx_flat[i], yy_flat[i]] for i in range(len(xx_flat))])


    # Classify the points in grid
    data = task2_hNN_AB(grid)
    data = data.reshape((x_values.shape[0], y_values.shape[0]))

    fig, ax = plt.subplots()
    plt.title('Decision Regions hNN AB')
    plt.xticks(np.arange(-2, 8, 1), fontsize=8)
    plt.yticks(np.arange(-3, 7, 1),fontsize=8)


    # Plot and safe image
    c = plt.contourf(x_values, y_values, data, cmap=cm.YlGnBu )
    proxy = np.array([plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in c.collections])
    plt.legend(proxy[[0,-1]], ['Class 0', 'Class 1'])

    plt.show()
    plt.draw()
    fig.savefig('t2_regions_hNN_AB.pdf')

def task2_sNN_AB(X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    
    # Hard coding precalculated weights for Layer 1
    weightAL1Matrix = np.array([[ 1.,-0.68871342, -4.63282829, 1.], [-0.17355183, 1., 0.52779144,-0.49109806], [-0.10824754, -0.75116169, 1.,0.2298064 ]])
    weightBL1Matrix = np.array([[ 1., -0.99634817, -2.69037292, 1.], [-0.41579484, 1., 1., -0.09038579],[ 0.41112314, -0.38886998, 0.18348208,-0.12973483]])

    weightL1Matrix = np.ones((weightAL1Matrix.shape[0], weightAL1Matrix.shape[1] + weightBL1Matrix.shape[1]))
    weightL1Matrix[:, [0,1,2,3]] = weightAL1Matrix[:]
    weightL1Matrix[:, [4,5,6,7]] = weightBL1Matrix[:]
    weightL1Matrix*=1000

    weightL2Matrix = np.ones((9,2))
    weightL2Matrix[:,0] = np.array([-3.9,1,1,1,1,0,0,0,0])*10
    weightL2Matrix[:,1] = np.array([-4.9, 0, 0, 0, 0, 2, 1, 1, 2])*10

    weightL3 = np.array([-0.5,-1,1]).reshape((3,1))
    weightL3*=200

    # Calculating output of First hidden layer
    L1 = np.zeros((X.shape[0], weightL1Matrix.shape[1]))

    for i in range(L1.shape[1]):
        L1[:,i] = task2_sNeuron(weightL1Matrix[:,i], X).reshape((X.shape[0],))
    
    # Calculating output of Second hidden layer
    L2 = np.zeros((X.shape[0], weightL2Matrix.shape[1]))

    for i in range(L2.shape[1]):
        L2[:,i] = task2_sNeuron(weightL2Matrix[:,i], L1).reshape((L2.shape[0],))

    # Calculating Final output
    Y = task2_sNeuron(weightL3, L2)
    # Setting threshold 0.5
    thresh = np.vectorize(lambda x: 1 if x>0.5 else 0)
    Y = thresh(Y)

    return Y

def task2_plot_regions_sNN_AB():
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


    plt.show()
    plt.draw()
    fig.savefig('t2_regions_sNN_AB.pdf')

def main():
    task2_plot_regions_hNN_A()
    task2_plot_regions_hNN_AB()
    task2_plot_regions_sNN_AB()


