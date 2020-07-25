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

def task1_mgc_cv(X, Y, CovKind, epsilon, Kfolds):
    # Input:
    #  X : N-by-D matrix of feature vectors (np.double)
    #  Y : N-by-1 label vector (np.int32)
    #  CovKind : scalar (np.int32)
    #  epsilon : scalar (np.double)
    #  Kfolds  : scalar (np.int32)
    #
    # Variables to save
    #  PMap   : N-by-1 vector of partition numbers (np.int32)
    #  Ms     : C-by-D matrix of mean vectors (np.double)
    #  Covs   : C-by-D-by-D array of covariance matrices (np.double)
    #  CM     : C-by-C confusion matrix (np.double)

    # INITIALIZING NECESSARY VALUES

    # number of classes and samples in Y
    classes, count = np.unique(Y, return_counts=True)
    idx = classes.argsort()[::1]
    classes = classes[idx]
    count = count[idx]
    num_classes = len(classes)

    # Data dimension
    D = X.shape[1]
    epsilon_matrix = np.zeros((D,D))
    np.fill_diagonal(epsilon_matrix, epsilon)
    # Number of samples of each class per partition
    # Accodring to class-wise partitioning algorithm
    Mc = (np.floor(count / Kfolds)).astype(int)
    PMap = np.zeros(np.size(Y, 0))

    # indexing the class of each sample
    Y_idx = [(idx, elem) for idx, elem in enumerate(Y)]

    Y_sep = dict((el, None) for el in classes)
    for idx, key in enumerate(Y_sep.keys()):
        # Seperating each class into Kfold arrays of indices for each partition
        Y_sep[key] = [id for (id, el) in Y_idx if el == int(key)]
        Y_sep[key] = [np.array(Y_sep[key][i:i+Mc[idx]]) for i in range(0, len(Y_sep[key]), int(Mc[idx]))]
        # Concatenating last to array for the last partition
        if len(Y_sep[key]) > Kfolds:
            Y_sep[key][-2] = np.concatenate((Y_sep[key][-2], Y_sep[key][-1]))
            Y_sep[key].pop()
        Y_sep[key] = np.array(Y_sep[key])

    
    # Filling in PMap
    for fold in range(1,Kfolds+1):
        for key in Y_sep.keys():
            for idx in Y_sep[key][fold-1]:
                PMap[idx] = fold
    PMap = PMap.astype(np.int32)

    par, count1 = np.unique(PMap, return_counts=True)
    idx = par.argsort()[::1]
    par = par[idx]
    count1 = count1[idx]

    # Calculating prior probability for each partition
    prior_p = dict((i, []) for i in range(1, Kfolds + 1))

    for p in prior_p.keys():
        for key in Y_sep.keys():
            prior_p[p].append(len(Y_sep[key][p-1]) / count1[p-1])
    
    
    # Holds the training data for each class for each partition
    partition = dict( (i,[]) for i in range(1,Kfolds+1))

    for p in partition.keys():
        for key in Y_sep.keys():
            idx = np.array([arr for i, arr in enumerate(Y_sep[key]) if i+1 != p])
            idx = np.array([elem for sublist in idx for elem in sublist])
            # Saving the indices of the training data of class 'key' for partition 'p'
            partition[p].append((key, idx))

    # Holds the mean and cov matrix for each class for each partition
    PMsCov = dict( (i,[]) for i in range(1,Kfolds+1))
    for p in PMsCov.keys():
        for i in range(len(Y_sep.keys())):
            Xm = X[partition[p][i][1], :]
            mean = MyMean(Xm)
            if CovKind == 1 or CovKind == 3: # Full matrix or Shared - adjusted later on
                cov = MyCov(Xm) + epsilon_matrix
                PMsCov[p].append((mean, cov))
            elif CovKind == 2: # In case of a diagonal matrix
                cov = MyCov(Xm)
                diag = np.diag(cov)
                cov_diag = np.zeros(cov.shape)
                np.fill_diagonal(cov_diag, diag)
                cov_diag = cov_diag + epsilon_matrix
                PMsCov[p].append((mean, cov_diag))

    if CovKind == 3: # Shared Matrix case
        for p in PMsCov.keys():
            covs = [ cov for (mean, cov) in PMsCov[p]]
            shared_cov = np.sum(covs, axis=0) / len(covs)
            for i in range(len(PMsCov[p])):
                # Assigning the shared matrix to all classes in the partition
                PMsCov[p][i]= (PMsCov[p][i][0], shared_cov)

    # Separating the means and cov-matrices
    PMs = dict((i,None) for i in range(1,Kfolds+1))
    PCov = dict((i,None) for i in range(1,Kfolds+1))
    for p in PMs.keys():
        PMs[p] = np.array([ mean for (mean, cov) in PMsCov[p]])
        PCov[p] = np.array([ cov for (mean, cov) in PMsCov[p]])

    # Holds the Confusion Matrix for every Partition
    PConfM = dict((i,None) for i in range(1,Kfolds+1))
    samples_num = np.zeros(Kfolds, dtype=np.int32)
    for p in PMs.keys():
        # Exctracting the test samples for partition p in Xp
        idx = np.where(PMap == p)[0]
        Xp = X[idx,:]
        samples_num[p-1] = len(idx) # Number of test samples in each partition
        # Holds the log posterior probability for all samples in Xp for each class
        log_pp_full = np.zeros((len(idx), num_classes))
        for i in range(len(prior_p[p])):
            # Calculating log_pp for each class
            mean = PMs[p][i]
            cov = PCov[p][i]
            log_pp = -0.5*np.diag((Xp - mean) @ np.linalg.inv(cov) @ (Xp - mean).T) - 0.5*np.log(np.linalg.det(cov)) + np.log(prior_p[p][i]) 
            log_pp_full[:,i] = log_pp
        
        # Assigning to the class with the highest log probability by taking the index
        classified = np.argmax(log_pp_full, axis=1)
        classified+=1
        classified = classified.reshape(-1,1)
        actual = Y[idx].reshape(-1,1)
        # Constructing the Confusion matrix from the test classification and the actual class
        conf_m = np.zeros((num_classes, num_classes), dtype=np.int32)
        for i in range(len(classified)):
            conf_m[actual[i][0]-1][classified[i][0]-1]+=1
        PConfM[p] = conf_m[:]
        # print(f'{np.sum(np.diag(conf_m))} / {len(idx)} = {np.sum(np.diag(conf_m))/len(idx)}')



    # Saving generated data

    scipy.io.savemat(f't1_mgc_{Kfolds}cv_PMap.mat', mdict={'PMap':PMap})
    for p in PMs.keys():
        Ms = PMs[p]
        Cov = PCov[p]
        CM = PConfM[p]
        scipy.io.savemat(f't1_mgc_{Kfolds}cv{p}_Ms.mat', mdict={'Ms':Ms})
        scipy.io.savemat(f't1_mgc_{Kfolds}cv{p}_ck{CovKind}_Covs.mat', mdict={'Covs': Cov})
        scipy.io.savemat(f't1_mgc_{Kfolds}cv{p}_ck{CovKind}_CM.mat', mdict={'CM':CM})

    # Calculating Final Confusion Matrix
    CM = 0
    for p in PConfM.keys():
        CM +=PConfM[p] / samples_num[p-1]
    CM = CM / Kfolds
    L = Kfolds + 1
    scipy.io.savemat(f't1_mgc_{Kfolds}cv{L}_ck{CovKind}_CM.mat', mdict={'CM':CM})


def main():
    data = scipy.io.loadmat('dset.mat')
    X = data['X'][:]
    Y_species = data['Y_species'][:]
    
    task1_mgc_cv(X, Y_species, CovKind = 1, epsilon = 0.01, Kfolds=5)
    task1_mgc_cv(X, Y_species, CovKind = 2, epsilon = 0.01, Kfolds=5)
    task1_mgc_cv(X, Y_species, CovKind = 3, epsilon = 0.01, Kfolds=5)
    task1_3(MyCov(X))

main()
