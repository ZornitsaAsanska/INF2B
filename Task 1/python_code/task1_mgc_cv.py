#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io
import math
import task1_1 as t1
import matplotlib.pyplot as plt

def PMap(X, Y, Kfolds):
    '''
    Dividing the data into partitions classwise
    Returns a N-by-1 vector holding the partition of each sample
    '''

    cl, num = np.unique(Y, return_counts=True)
    idx = cl.argsort()[::1]
    cl = cl[idx]
    num = num[idx]

    Mc = np.floor(num / Kfolds).astype(int)
    PMap = np.zeros(np.size(Y, 0))

    # Dictionary to hold the indices of each class in each partition
    separation = dict((el, None) for el in cl)
    for idx, key in enumerate(separation.keys()):
        # Seperating each class into Kfold arrays of indices for each partition
        class_idx = np.argwhere(Y == int(key)).reshape(-1,)
        separation[key] = [np.array(class_idx[i:i+Mc[idx]]) for i in range(0, len(class_idx), Mc[idx])]
        # Concatenating last to array for the last partition
        if len(separation[key]) > Kfolds:
            separation[key][-2] = np.concatenate((separation[key][-2], separation[key][-1]))
            separation[key].pop()
        separation[key] = np.array(separation[key])
    
    # Filling in PMap
    for fold in range(1, Kfolds + 1):
        for key in separation.keys():
            for idx in separation[key][fold-1]:
                PMap[idx] = fold
    PMap = PMap.astype(np.int32)
    return PMap

def prior_probability(Y, PMap, Kfolds):
    '''
    Returns a dictionary with the partitions as keys and
    array of the prior probability of each class in each partition
    '''

    cl, num1 = np.unique(Y, return_counts=True)
    idx = cl.argsort()[::1]
    cl = cl[idx]
    num1 = num1[idx]

    par, num2 = np.unique(PMap, return_counts=True)
    idx = par.argsort()[::1]
    par = par[idx]
    num2 = num2[idx]

    prior_p = dict((i, []) for i in range(1, Kfolds + 1))

    for p in prior_p.keys():
        for c in cl:
            idx = np.argwhere(Y == c).reshape(-1,)
            part_p_c = np.argwhere(PMap[idx] == p).reshape(-1,)
            prior_p[p].append(len(part_p_c) / num2[p-1])

    return prior_p

def Mean_and_Cov(X, Y, CovKind, PMap, Kfolds, epsilon):
    '''
    Returns two dictionaries of 
    the Mean values and Covariance matrices for each partition
    The means and matrices for each class are held within an array as dictionary values
    '''
    D = X.shape[1]
    epsilon_matrix = np.zeros((D,D))
    np.fill_diagonal(epsilon_matrix, epsilon)

    cl = np.unique(Y)
    idx = cl.argsort()[::1]
    cl = cl[idx]

    indexed_PMap = [(idx, part) for idx, part in enumerate(PMap)]

    # Holds the training data for each class for each partition
    training_data = dict( (i, []) for i in range(1, Kfolds + 1))

    for p in training_data.keys():
        for c in cl:
            class_c = np.argwhere(Y == c).reshape(-1,)
            train_indices = [idx for (idx, part) in indexed_PMap if idx in class_c and part != p]
            training_data[p].append((c, train_indices))

    PMs = dict((i,[]) for i in range(1, Kfolds + 1))
    PCov = dict((i,[]) for i in range(1, Kfolds + 1))

    for p in PMs.keys():
        for i in range(len(cl)):
            # indices of training data of partition 'p' for class 'c'
            Xm = X[training_data[p][i][1], :]
            mean = t1.MyMean(Xm)
            PMs[p].append(mean)

            if CovKind == 1 or CovKind == 3: # Full matrix or Shared - adjusted later on
                cov = t1.MyCov(Xm) + epsilon_matrix
                PCov[p].append(cov)
            elif CovKind == 2: # In case of a diagonal matrix
                cov = t1.MyCov(Xm)
                diag = np.diag(cov)
                cov_diag = np.zeros(cov.shape)
                np.fill_diagonal(cov_diag, diag)
                cov_diag = cov_diag + epsilon_matrix
                PCov[p].append(cov_diag)

    if CovKind == 3: # Shared Matrix case
        for p in PCov.keys():
            covs = [cov for cov in PCov[p]]
            shared_cov = np.sum(covs, axis = 0) / len(covs)
            for i in range(len(cl)):
                # Assigning the shared matrix to all classes in the partition
                PCov[p][i] = shared_cov

    for p in PMs.keys():
        PMs[p] = np.array(PMs[p])
        PCov[p] = np.array(PCov[p])

    return PMs, PCov


def confusion_M(X, Y, PMs, PCov, PMap, Kfolds, prior_p):
    '''
    Conducts classification experiment on each partition
    Returns confusion matrix for each partition
    Returns final confusion matrix with accuracy rate for the whole experiment
    '''
    cl = np.unique(Y)
    idx = cl.argsort()[::1]
    cl = cl[idx]

    PConfM = dict((i, None) for i in range(1, Kfolds + 1))
    samples_num = np.zeros(Kfolds, dtype=np.int32)
    for p in PMs.keys():
        idx = np.where(PMap == p)[0]
        Xp = X[idx, :]
        samples_num[p-1] = len(idx) # Number of test samples in each partition
        # Holds the log posterior probability for all samples in Xp for each class
        log_pp_full = np.zeros((len(idx), len(cl)))

        for i in range(len(cl)):
            # Calculating log_pp for each class
            mean = PMs[p][i]
            cov = PCov[p][i]
            log_pp = -0.5*np.diag((Xp - mean) @ np.linalg.inv(cov) @ (Xp - mean).T) - 0.5*np.log(np.linalg.det(cov)) + np.log(prior_p[p][i])
            log_pp_full[:, i] = log_pp[:]
        
        # Assigning to the class with the highest log probability by taking the index
        classified = np.argmax(log_pp_full, axis=1)
        classified+=1
        classified = classified.reshape(-1,1)
        actual = Y[idx].reshape(-1,1)
        # Constructing the Confusion matrix from the test classification and the actual class
        conf_m = np.zeros((len(cl), len(cl)), dtype=np.int32)
        for i in range(len(classified)):
            conf_m[actual[i][0]-1][classified[i][0]-1]+=1
        PConfM[p] = conf_m[:]

    # Calculating Final Confusion Matrix
    Final_ConfM = np.zeros((len(cl), len(cl)))
    for p in PConfM.keys():
        Final_ConfM +=PConfM[p] / samples_num[p-1]
    Final_ConfM = Final_ConfM / Kfolds

    return PConfM, Final_ConfM


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

    pMap = PMap(X, Y, Kfolds)
    prior_p = prior_probability(Y, pMap, Kfolds)
    PMs, PCov = Mean_and_Cov(X, Y, CovKind, pMap, Kfolds, epsilon)
    PConfM, Final_ConfM = confusion_M(X, Y, PMs, PCov, pMap, Kfolds, prior_p)

    # Saving generated data

    scipy.io.savemat(f't1_mgc_{Kfolds}cv_PMap.mat', mdict={'PMap':pMap})
    for p in PMs.keys():
        Ms = PMs[p]
        Cov = PCov[p]
        CM = PConfM[p]
        scipy.io.savemat(f't1_mgc_{Kfolds}cv{p}_Ms.mat', mdict={'Ms':Ms})
        scipy.io.savemat(f't1_mgc_{Kfolds}cv{p}_ck{CovKind}_Covs.mat', mdict={'Covs': Cov})
        scipy.io.savemat(f't1_mgc_{Kfolds}cv{p}_ck{CovKind}_CM.mat', mdict={'CM':CM})

    L = Kfolds + 1
    scipy.io.savemat(f't1_mgc_{Kfolds}cv{L}_ck{CovKind}_CM.mat', mdict={'CM':Final_ConfM})
    return Final_ConfM


def epsilonAccuracy():

    data = scipy.io.loadmat('dset.mat')
    X = data['X'][:]
    Y_species = data['Y_species'][:]

    epsilon = np.arange(0.01,1,step=0.01)
    accuracy = np.zeros(epsilon.shape)
    for idx, e in enumerate(epsilon):
        accuracy[idx] = np.sum( np.diag(task1_mgc_cv(X, Y_species, CovKind=1, epsilon = e, Kfolds=5)))
    fig, ax = plt.subplots()
    plt.plot(epsilon, accuracy)

    plt.title('Accuracy wrt Epsilon')
    plt.xlabel('Epsilon Value')
    plt.ylabel('Accuracy of classification')
    plt.show()
    plt.draw()
    fig.savefig('Epsilon Accuracy1.png')

    
def main():
    data = scipy.io.loadmat('dset.mat')
    X = data['X'][:]
    Y_species = data['Y_species'][:]
    Y_species = Y_species.reshape(-1,)

    CM1 = task1_mgc_cv(X, Y_species, CovKind=1, epsilon=0.01, Kfolds=5)
    CM2 = task1_mgc_cv(X, Y_species, CovKind=2, epsilon=0.01, Kfolds=5)
    CM3 = task1_mgc_cv(X, Y_species, CovKind=3, epsilon=0.01, Kfolds=5)
    acc1 = np.sum(np.diag(CM1))
    acc2 = np.sum(np.diag(CM2))
    acc3 = np.sum(np.diag(CM3))
    
    fig, ax = plt.subplots()
    objects = ['Full Matrix', 'Diagonal Matrix', 'Shared Matrix']
    y_pos = np.arange(len(objects))
    values = [acc1, acc2, acc3]
    labels = [round(i*0.1, 1) for i in range(10)]

    ax.bar(y_pos, values, align='center', alpha=0.5, color = 'b')

    ax.set_yticks(labels)
    ax.set_yticklabels(labels)
    plt.xticks(y_pos, objects)
    plt.title('Classification Accuracy, epsilon=0.01')
    plt.ylabel('Classification Accuracy')

    plt.show()
    plt.draw()
    fig.savefig('Classification Accuracy 0.01.png')

    print([acc1, acc2, acc3])

main()

