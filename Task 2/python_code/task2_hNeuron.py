#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io

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
