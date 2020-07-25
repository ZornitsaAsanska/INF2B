#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm as cm

def task1_1(X, Y):
    # Input:
    #  X : N-by-D data matrix (np.double)
    #  Y : N-by-1 label vector (np.int32)
    # Variables to save
    #  S : D-by-D covariance matrix (np.double) to save as 't1_S.mat'
    #  R : D-by-D correlation matrix (np.double) to save as 't1_R.mat'

    S = MyCov(X)
    R = MyCorr(X)


    scipy.io.savemat('t1_S.mat', mdict={'S': S})
    scipy.io.savemat('t1_R.mat', mdict={'R': R})

def MyMean(X):
    # Equivalent to np.mean()
    mean = np.sum(X, axis=0)/np.size(X,0)
    return mean

def MyCov(X):
    # Covariance matrix using MLE
    mu = MyMean(X)
    covM = X - np.tile(mu, (np.size(X,0),1))
    covM = np.matmul(np.transpose(covM), covM) / np.size(X,0)
    return covM

def MyCorr(X):
    # Correlation matrix using MLE
    covM = MyCov(X)
    diag = covM.diagonal().reshape(-1,1)
    corrM = np.divide(covM, np.sqrt(np.matmul(diag, diag.T)))
    return corrM

def plot(m):
    # Plotting the Correlation matrix m given as parameter

    frame = pd.DataFrame(data = m)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    cmap = cm.get_cmap('jet')
    # Plotting the matrix
    cax = ax.imshow(frame, interpolation='nearest', cmap = cmap)
    ax.grid(True)
    # Assigning labels
    plt.title('Correlation Matrix')
    labels = [i for i in range(m.shape[0])]
    ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    ax.set_yticks(labels)
    ax.set_yticklabels(labels)

    # Showing and saving plot
    plt.show()
    plt.draw()
    fig.savefig('Correlation Matrix.png')



