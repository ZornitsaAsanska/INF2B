#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io
import task1_1
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def task1_3(Cov):
    # Input:
    #  Cov : D-by-D covariance matrix (np.double)
    # Variales to save:
    #  EVecs : D-by-D matrix of column vectors of eigen vectors (np.double)  
    #  EVals : D-by-1 vector of eigen values (np.double)  
    #  Cumvar : D-by-1 vector of cumulative variance (np.double)  
    #  MinDims : 4-by-1 vector (np.int32)

    EVals, EVecs = np.linalg.eig(Cov)
    # Aranging eigenvalues in descending order
    idx = EVals.argsort()[::-1]
    EVals = EVals[idx]
    EVecs = EVecs[:, idx]

    # Calculating Cumulative variance
    Cumvar = np.cumsum(EVals)
    CumvarNorm = (Cumvar / Cumvar[-1])*100
    # Making first value of Eigenvectors positive
    for i in range(np.size(EVecs,1)):
        if EVecs[0, i] < 0:
            EVecs[:,i]*=(-1)

    MinDims = np.zeros(4)

    for i in range(len(CumvarNorm)):
        if CumvarNorm[len(CumvarNorm)-1-i] > 95:
            MinDims[3] = len(CumvarNorm)-i
        elif CumvarNorm[len(CumvarNorm)-1-i] > 90:
            MinDims[2] = len(CumvarNorm)-i
        elif CumvarNorm[len(CumvarNorm)-1-i] > 80:
            MinDims[1] = len(CumvarNorm)-i
        elif CumvarNorm[len(CumvarNorm)-1-i] > 70:
            MinDims[0] = len(CumvarNorm)-i

    scipy.io.savemat('t1_EVecs.mat', mdict={'EVecs': EVecs})
    scipy.io.savemat('t1_EVals.mat', mdict={'EVals': EVals})
    scipy.io.savemat('t1_Cumvar.mat', mdict={'Cumvar': Cumvar})
    scipy.io.savemat('t1_MinDims.mat', mdict={'MinDims': MinDims})


def plotPCA(X, EVecs, Y_species):

    unique, count = np.unique(Y_species, return_counts=True)
    idx = unique.argsort()[::1]
    unique = unique[idx]
    count = count[idx]
    Y1f = X @ EVecs[:,0]
    Y2f = X @ EVecs[:,1]
    classes = ['Leptodactylus fuscus', 'Adenomera andreae', 'Adenomera hylaedactyla', 'Hyla minuta', 'Scinax ruber', 'Osteocephalus oophagus', 'Hypsiboas cinerascens', 'Hypsiboas cordobae', 'Rhinella granulosa', 'Ameerega trivittata']

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    scatter = plt.scatter(Y1f, Y2f, c=Y_species, cmap=cm.viridis, s=3, marker = 'x')

    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.xlim((np.amin(Y1f)*1.1, np.amax(Y1f)*1.1))
    plt.ylim((np.amin(Y2f)*1.1, np.amax(Y2f)*1.1))

    plt.title('2D PCA Plot')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')

    plt.show()
    plt.draw()
    fig.savefig(f'2D PCA Plot.png')

def plotCumVar(CumVar, X):
    fig, ax = plt.subplots()
    fig.canvas.draw()
    labels = np.arange(0,X.shape[0], 2)
    ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    plt.plot(CumVar)
    plt.title('Cumulative Variance')
    plt.show()
    plt.draw()
    fig.savefig("CumVar.png")








