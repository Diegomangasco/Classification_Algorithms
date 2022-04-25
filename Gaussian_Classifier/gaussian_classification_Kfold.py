import scipy
import scipy.special
import sklearn.datasets
import numpy
import sys

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

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
    
    K = int(sys.argv[1])
    # Check if the parameter is consistent with our database
    if(K > D.shape[1] or K < 2 or 150%K != 0):
        return 0 
    
    # Split the database in K partitions both for data and labels
    K_fold_set = numpy.hsplit(D, K)
    K_fold_set = numpy.array(K_fold_set)
    K_fold_labels = numpy.array(L)
    K_fold_labels = numpy.split(K_fold_labels, K)
    t = 0
    selector = [x for x in range(0, K) if x!=t]

    accuracy_list = []
    error_list = []
    
    # For cycle for doing K fold cross validation 
    for i in range(0, K):
        K_validation_set = K_fold_set[i]
        K_validation_label_set = K_fold_labels[i]
        t = i
        K_training_set = K_fold_set[selector]
        K_training_set_def = K_training_set[0]
        K_training_labels_set_def = K_fold_labels[0]
        # Concatenate the arrays for having one training set and one test set
        for j in range(1, K-1):
            K_training_set_def = numpy.concatenate((K_training_set_def, K_training_set[j]), axis=1)
            K_training_labels_set_def = numpy.concatenate((K_training_labels_set_def, K_fold_labels[j]), axis=0)
            
        # Calculate mean and covariance for each class of the dataset
        class_mean1, covariance_matrix1 = calculate_parameters(0, K_training_set_def, K_training_labels_set_def)
        class_mean2, covariance_matrix2 = calculate_parameters(1, K_training_set_def, K_training_labels_set_def)
        class_mean3, covariance_matrix3 = calculate_parameters(2, K_training_set_def, K_training_labels_set_def)
         # Calculate the likelihood for all the test set with the mean and covariance of each class
        S = wrapper_logpdf(K_validation_set, class_mean1, covariance_matrix1)
        S = numpy.concatenate((S, wrapper_logpdf(K_validation_set, class_mean2, covariance_matrix2)), axis=1)
        S = numpy.concatenate((S, wrapper_logpdf(K_validation_set, class_mean3, covariance_matrix3)), axis=1)
        S = S.T
        Pc = 1/3    # Default value for prior probability of a class
        S_joint = S*Pc  # Joint distribution for samples and classes
        S_sum = S_joint.sum(0) # For each sample we sum the Joint distribution for the three classes (sum the rows) 
        S_marginal = numpy.reshape(S_sum, (1, S_sum.shape[0]))  # Reshape is needed because the sum operator puts the results in a column array
        S_post = S_joint/S_marginal # Posterior probability -> Joint distribution over Marginal distribution
        # For each column (sample) we search the maximum probability that indicates the class predicted 
        # (classes are the rows of the matrix)
        Predicted_labels = numpy.argmax(S_post, axis=0) 
        # Accuracy and Error rate
        accuracy_array = []
        for i in range(0, K_validation_label_set.size):
            if Predicted_labels[i] == K_validation_label_set[i]:
                accuracy_array.append(1)
            else:
                accuracy_array.append(0)
        accuracy = numpy.array(accuracy_array).sum(0)/len(accuracy_array)
        error_rate = 1 - accuracy
        accuracy_list.append(accuracy)
        error_list.append(error_rate)

    accuracy_mean = (sum(accuracy_list)/K)*100
    error_mean = (sum(error_list)/K)*100
    print(accuracy_mean)
    print(error_mean)

main()
