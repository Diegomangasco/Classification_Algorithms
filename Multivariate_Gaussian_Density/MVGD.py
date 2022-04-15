from math import pi
import numpy as npy
import matplotlib.pyplot as plt

def vrow(v):
    return v.reshape((1, v.size))

def calculate_mean_covariance(data):
    data = npy.array(data)
    mean = data.mean(1)
    centeredData = data - npy.reshape(mean, (mean.size, 1))
    covarianceMatrix = npy.dot(centeredData, centeredData.T)/data.shape[1]
    return mean, covarianceMatrix

def logpdf_GAU_ND(X, mu, C):
    M = mu.shape[0]
    Xc = (X-mu)
    P = npy.linalg.inv(C)
    const = -(M/2)*npy.log(2*npy.pi);
    const += -0.5*npy.linalg.slogdet(C)[1]
    result = -0.5*npy.dot(Xc.T, npy.dot(P, Xc))
    result += const 
    return result

def wrapper_logpdf(X, mu, C):
    return_value = []
    for i in range(0, X.size):
        return_value.append(logpdf_GAU_ND(X[:, i], mu, C))
    return return_value

def loglikelihood(X1D, m_ML, C_ML):
    res = 0
    for i in range(0, X1D.size):
        res += logpdf_GAU_ND(X1D[:, i], m_ML, C_ML)
    return res

def main():
    X1D = npy.load("X1D.npy")
    res = calculate_mean_covariance(X1D)
    m_ML = res[0]
    C_ML = res[1]
    X1D_sort = npy.sort(X1D)
    plt.figure()
    plt.hist(X1D_sort.ravel(), bins=50, density=True)
    plt.show()
    XPlot = npy.linspace(-8, 12, 1000)
    plt.plot(X1D_sort.ravel(), npy.exp(wrapper_logpdf(X1D_sort, m_ML, C_ML)))
    plt.show()
    ll = loglikelihood(X1D, m_ML, C_ML)
    print(ll)

main()