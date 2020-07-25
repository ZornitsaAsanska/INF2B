#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def task2_sNeuron(W, X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    #  W : (D+1)-by-1 vector of weights (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    Xext = np.ones((X.shape[0], X.shape[1] + 1))
    Xext[:,[i for i in range(1, X.shape[1]+1)]] = X
    sigmoid_param = Xext @ W
    # sigmoid_param = np.asarray(sigmoid_param, dtype=np.float128)
    sigmoid = np.vectorize(lambda x: 1/(1+np.exp(-x)))
    Y = sigmoid(sigmoid_param)
    Y = Y.reshape(-1,1)
    return Y

