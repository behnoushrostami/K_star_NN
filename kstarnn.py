import distance
import numpy as np

def kstarnn(x_train, y_train, x_test, y_test, L_C, distance_metric):
    """
    A function that computes the k*-nearest neighbor estimator given x_train, y_train, L_C and returns the optimal number
    of nearest neighbors as well as the absolute deviation error from the ground truth per decision point given x_test
    and y_test.
    :param x_train: training data set (n x d)
    :param y_train: training data output variable (n x 1)
    :param x_test: test data set (m x d)
    :param y_test: test data output variable (m x 1)
    :param L_C: Lipschitz to noise ratio
    :param distance_metric: distance metric to use, see distances.py for options
    :return (errors, kstars, acc): (absolute deviation error of estimator from ground truth (m x 1), optimal number of
    nearest neighbors per decision point (m x 1), prediction accuracy (m x 1))
    """
    m = x_test.shape[0]
    n = x_train.shape[0]
    estimates = np.zeros(m,)
    kstars = np.zeros(m,)
    # Initialize Ordered Distances
    distances = distance.get_distances(x_train, x_test, distance_metric)
    beta = L_C * distances
    beta.sort(axis=1)
    # Iterate over test data
    for j in range(0, m):
        # Initialize lambda
        k = 0
        lmbda = beta[j, 0] + 1
        # Find optimal lambda
        while lmbda > beta[j, k] and k < n - 1:
            k = k + 1
            lmbda = (1 / k) * (np.sum(beta[j, :k]) + np.sqrt(k + (np.sum(beta[j, :k]) ** 2) - k * np.sum(beta[j, :k] ** 2)))
        # Store locally adaptive and optimal number of nearest neighbors for test point j
        kstars[j] = k
        # Compute k*-nn estimate
        den = np.sum((lmbda - L_C * distances[j, :]) * (L_C * distances[j, :] < lmbda))
        alpha = ((lmbda - L_C * distances[j, :]) * (L_C * distances[j, :] < lmbda)) / den
        estimates[j] = np.sum(alpha * y_train)
    # Return absolute deviation error per test point, number of nearest neighbors per test point
    errors = np.abs(estimates - y_test)
    
    # Assign label to the closest class
    m = x_test.shape[0]
    label = np.unique(y_train)
    deltaFx = np.abs(np.reshape(estimates, (m, 1)) - label)
    y_est = np.argmin(deltaFx, axis=1)
    
    # Prediction accuracy
    acc = np.sum(y_est == y_test) / m      
    return errors, kstars, acc
