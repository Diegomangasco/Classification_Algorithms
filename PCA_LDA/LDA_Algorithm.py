import sys
from unittest import result
import numpy as npy
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.linalg

def mcol(v):
    return v.reshape((v.size, 1))

def loadFile(arg):
    file = open(arg, 'r')
    categories = []
    Dlist = []
    hlabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    for line in file.readlines():
        try:
            words = line.split(",")[0:4]
            attrs = mcol(npy.array([float(i) for i in words]))
            name = line.split(',')[-1].strip()
            label = hlabels[name]
            categories.append(label)
            Dlist.append(attrs)
        except:
            pass

    return npy.hstack(Dlist), npy.array(categories, dtype=npy.int32)

def computeCovarianceForEachClass(matrix, categories):
    data = npy.array(matrix)
    D1 = data[:, categories==0]
    D2 = data[:, categories==1]
    D3 = data[:, categories==2]
    mu1 = D1.mean(1)
    mu2 = D2.mean(1)
    mu3 = D3.mean(1)
    CD1 = D1 - mcol(mu1)
    CD2 = D2 - mcol(mu2)
    CD3 = D3 - mcol(mu3)
    CM1 = npy.dot(CD1, CD1.T)
    CM2 = npy.dot(CD2, CD2.T)
    CM3 = npy.dot(CD3, CD3.T)
    result = (CM1 + CM2 + CM3)/data.shape[1]
    return result

def computeCovarianceBetweenClasses(matrix, categories):
    data = npy.array(matrix)
    D1 = data[:, categories==0]
    D2 = data[:, categories==1]
    D3 = data[:, categories==2]
    mu1 = D1.mean(1)
    mu2 = D2.mean(1)
    mu3 = D3.mean(1)
    mu = data.mean(1)
    dataForEachClass = data.shape[1]/3
    mean1 = dataForEachClass * npy.dot((mcol(mu1)-mcol(mu)), (mcol(mu1)-mcol(mu)).T)
    mean2 = dataForEachClass * npy.dot((mcol(mu2)-mcol(mu)), (mcol(mu2)-mcol(mu)).T)
    mean3 = dataForEachClass * npy.dot((mcol(mu3)-mcol(mu)), (mcol(mu3)-mcol(mu)).T)
    result = (mean1 + mean2 + mean3)/data.shape[1]
    return result 

def plotLDA(U, data, categories):
    DP = npy.dot(U.T, data)
    D0 = DP[:, categories==0]
    D1 = DP[:, categories==1]
    D2 = DP[:, categories==2]
    plt.figure()
    plt.scatter(D0[0, :], D0[1, :], label = 'Setosa')
    plt.scatter(D1[0, :], D1[1, :], label = 'Versicolor')
    plt.scatter(D2[0, :], D2[1, :], label = 'Virginica')
    plt.legend()
    plt.show()

def main():
    arg = sys.argv[1]
    returnValues = loadFile(arg)
    covarianceWithinClasses = computeCovarianceForEachClass(returnValues[0], returnValues[1])
    covarianceBetweenClasses = computeCovarianceBetweenClasses(returnValues[0], returnValues[1])
    s, U = scipy.linalg.eigh(covarianceBetweenClasses, covarianceWithinClasses)
    m = 2
    W = U[:, ::-1][:, 0:m]
    UW, _, _ = npy.linalg.svd(W)
    U = UW[:, 0:m]
    print(W)
    print(U)
    plotLDA(U, returnValues[0], returnValues[1])

main()
