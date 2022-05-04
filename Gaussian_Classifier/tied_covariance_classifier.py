import scipy
import scipy.special
import sklearn.datasets
import numpy

def class_columns(class_identifier, training_data, training_labels):
    return training_data[:, training_labels == class_identifier].shape[1]

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

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

def calculate_parameters(class_number, dataset, categories):
    D = dataset[:, categories==class_number]
    mean = D.mean(1)
    covariance = numpy.dot((D-mean.reshape(mean.size, 1)), (D-mean.reshape(mean.size, 1)).T)/D.shape[1]
    # Nc is not the number of classes, but the number of samples for a specific class!
    return numpy.array(mean), numpy.array(covariance)

def logpdf_GAU_ND(X, mu, C):
    M = mu.shape[0]
    Xc = (X-mu)
    P = numpy.linalg.inv(C)
    const = -(M/2)*numpy.log(2*numpy.pi);
    const += -0.5*numpy.linalg.slogdet(C)[1]
    result = -0.5*numpy.dot(Xc.T, numpy.dot(P, Xc))
    result += const 
    return result

def wrapper_logpdf(X, mu, C):
    return_value = []
    for i in range(0, X.shape[1]):
        return_value.append(logpdf_GAU_ND(X[:, i], mu, C))
    return_value = numpy.exp(return_value)
    return numpy.reshape(return_value, (return_value.size, 1))

def wrapper_logpdf_with_log_densities(X, mu, C):
    return_value = []
    for i in range(0, X.shape[1]):
        return_value.append(logpdf_GAU_ND(X[:, i], mu, C))
    return numpy.reshape(return_value, (len(return_value), 1))

def main():
    D, L = load_iris()
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    # DTR and DTE have 4 rows, one for each variable considered 
    # The number of classes is 3
    
    # Calculate mean and covariance for each class of the dataset
    class_mean1, covariance_matrix1 = calculate_parameters(0, DTR, LTR)
    class_mean2, covariance_matrix2 = calculate_parameters(1, DTR, LTR)
    class_mean3, covariance_matrix3 = calculate_parameters(2, DTR, LTR)

    # Tied covariance matrix
    N_1 = class_columns(0, DTR, LTR)
    N_2 = class_columns(1, DTR, LTR)
    N_3 = class_columns(2, DTR, LTR)
    tied_covariance = (1/DTR.shape[1])*(covariance_matrix1*N_1 + covariance_matrix2*N2 + covariance_matrix3*N3)
    print(tied_covariance)

    # Calculate the likelihood for all the test set with the mean and covariance of each class
    S = wrapper_logpdf(DTE, class_mean1, tied_covariance)
    S = numpy.concatenate((S, wrapper_logpdf(DTE, class_mean2, tied_covariance)), axis=1)
    S = numpy.concatenate((S, wrapper_logpdf(DTE, class_mean3, tied_covariance)), axis=1)
    S = S.T

    Pc = 1/3    # Default value for prior probability of a class
    S_joint = S*Pc  # Joint distribution for samples and classes
    
    S_sum = S_joint.sum(0) # For each sample we sum the Joint distribution for the three classes (sum the rows) 
    S_marginal = numpy.reshape(S_sum, (1, S_sum.shape[0]))  #Reshape is needed because the sum operator puts the results in a column array
    S_post = S_joint/S_marginal # Posterior probability -> Joint distribution over Marginal distribution
    #For each column (sample) we search the maximum probability that indicates the class predicted 
    # (classes are the rows of the matrix)
    Predicted_labels = numpy.argmax(S_post, axis=0) 
    
    # Accuracy and Error rate
    accuracy_array = []
    for i in range(0, LTE.size):
        if Predicted_labels[i] == LTE[i]:
            accuracy_array.append(1)
        else:
            accuracy_array.append(0)

    accuracy = numpy.array(accuracy_array).sum(0)/len(accuracy_array)
    error_rate = 1 - accuracy
    print(accuracy)
    print(error_rate)


    # REPEAT THE PROCEDURE WITH LOGARITHM DENSITIES
    # The formulas are the same as before but they are adapted to the logarithm domain

def main2():
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    
    class_mean1, covariance_matrix1 = calculate_parameters(0, DTR, LTR)
    class_mean2, covariance_matrix2 = calculate_parameters(1, DTR, LTR)
    class_mean3, covariance_matrix3 = calculate_parameters(2, DTR, LTR)

    # Tied covariance matrix
    # Tied covariance matrix
    N_1 = class_columns(0, DTR, LTR)
    N_2 = class_columns(1, DTR, LTR)
    N_3 = class_columns(2, DTR, LTR)
    tied_covariance = (1/DTR.shape[1])*(covariance_matrix1*N_1 + covariance_matrix2*N2 + covariance_matrix3*N3)
    print(tied_covariance)

    S = wrapper_logpdf_with_log_densities(DTE, class_mean1, tied_covariance)
    S = numpy.concatenate((S, wrapper_logpdf_with_log_densities(DTE, class_mean2, tied_covariance)), axis=1)
    S = numpy.concatenate((S, wrapper_logpdf_with_log_densities(DTE, class_mean3, tied_covariance)), axis=1)
    S = S.T

    Pc = numpy.log(1/3)
    Log_joint = S + Pc

    # The function logsumexp from scipy library permits to implement the log-sum-exp trick
    Log_marginal = numpy.reshape(scipy.special.logsumexp(Log_joint, axis=0), (1, Log_joint.shape[1]))
    Log_posterior_probability = Log_joint - Log_marginal
    SPost = numpy.exp(Log_posterior_probability)
   
    Predicted_labels = numpy.argmax(SPost, axis=0)

    accuracy_array = []
    for i in range(0, LTE.size):
        if Predicted_labels[i] == LTE[i]:
            accuracy_array.append(1)
        else:
            accuracy_array.append(0)

    accuracy = numpy.array(accuracy_array).sum(0)/len(accuracy_array)
    error_rate = 1 - accuracy
    print(accuracy)
    print(error_rate)

main()
main2()

# The two results are the same
