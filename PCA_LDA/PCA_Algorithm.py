from cProfile import label
import numpy 
import matplotlib
import matplotlib.pyplot as plt
import sys

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
            attrs = mcol(numpy.array([float(i) for i in words]))
            name = line.split(',')[-1].strip()
            label = hlabels[name]
            categories.append(label)
            Dlist.append(attrs)
        except:
            pass

    return numpy.hstack(Dlist), numpy.array(categories, dtype=numpy.int32)

def computePCA(matrix):
    data = numpy.array(matrix)
    mu = data.mean(1)
    centeredData = data - numpy.reshape(mu, (mu.size, 1))
    covarianceMatrix = numpy.dot(centeredData, centeredData.T)/data.shape[1]
    s, U = numpy.linalg.eigh(covarianceMatrix)
    m = 2
    P = U[:, ::-1][:, 0:m]  #take all the columns in the reverse order (-1), and then takes only the first m columns
    DP = numpy.dot(P.T, data)
    #DProjList = []
    #for i in range(data.shape[1]):
    #    xi = mcol(data[:, i])
    #    yi = numpy.dot(P.T, xi)
    #    DProjList.append(yi)
    #DY = numpy.hstack(DProjList)
    return DP

def scatterPlot(DP, categories):
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
    matrix = returnValues[0]
    categories = returnValues[1]
    result = computePCA(matrix)
    scatterPlot(result, categories)

main()
