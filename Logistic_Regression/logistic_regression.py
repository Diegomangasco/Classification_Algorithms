import numpy
import scipy
import sklearn
import numpy.linalg
import scipy.optimize
import sklearn.datasets

class logRegClass:
    def __init__(self, DTR, LTR, l):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
    def logreg_obj(self, v):
        w, b = v[0:-1], v[-1]
        w = numpy.reshape(w, (1, w.size))
        constant = (self.l/2)*(numpy.linalg.norm(w)**2)
        counter = 0
        log_vector = []
        for i in range(0, len(self.LTR)):
            sample = numpy.reshape(self.DTR[:, i], (self.DTR.shape[0], 1))
            S = numpy.dot(w, sample) + b
            if self.LTR[i] == 0:
                zi = -1
            else:
                zi = 1
            log_vector.append(numpy.logaddexp(0, -zi*S))
            counter += 1
        return constant + (1/counter)*numpy.sum(log_vector)

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L


D, L = load_iris_binary()
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
l = 0.001;
logRegObj = logRegClass(DTR, LTR, l)
x0 = numpy.zeros(DTR.shape[0] + 1)
# You can now use logRegObj.logreg_obj as objective function:
minimum_position, function_value, dictionary = scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, x0, maxfun=20000, maxiter=20000, approx_grad=True)
w_for_minimum = minimum_position[0:-1]
b_for_minimum = minimum_position[-1]
# Compute S 
w_for_minimum = numpy.reshape(w_for_minimum, (1, w_for_minimum.shape[0]))
S = numpy.dot(w_for_minimum, DTE) + b_for_minimum
# Predicted labels

Predicted_labels = []
for i in range(0, S.shape[1]):
    if S[0][i] > 0:
        Predicted_labels.append(1)
    else:
        Predicted_labels.append(0)

accuracy_array = []
for j in range(0, len(LTE)):
    if Predicted_labels[j] == LTE[j]:
        accuracy_array.append(1)
    else:
        accuracy_array.append(0)

accuracy = sum(accuracy_array)/len(accuracy_array)
error_rate = 1 - accuracy

print(accuracy)
print(error_rate)
