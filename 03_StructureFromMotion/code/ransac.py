from lse import least_squares_estimation
import numpy as np

def ransac_estimator(X1, X2, num_iterations=60000):
    sample_size = 8

    eps = 10**-4

    best_num_inliers = -1
    best_inliers = None
    best_E = None

    for i in range(num_iterations):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]

        """ YOUR CODE HERE
        """
        E = least_squares_estimation(X1[sample_indices,:], X2[sample_indices,:])
        Skewz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        inliers = sample_indices

        for j in test_indices:
            #Calculate distances
            d2 = (X2[j,:] @ (E@X1[j,:].T))**2/(np.linalg.norm(Skewz@(E@X1[j,:]))**2)
            d1 = (X2[j,:] @ (E@X1[j, :].T))**2/(np.linalg.norm(Skewz@E.T@X2[j, :])**2)

            #Check if inliers
            if d1+d2 < eps:
                inliers = np.append(inliers, j)
            

        """ END YOUR CODE
        """
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            best_inliers = inliers


    return best_E, best_inliers