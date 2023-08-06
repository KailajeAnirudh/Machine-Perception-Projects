import numpy as np


def est_homography(X, Y):
    """
    Calculates the homography of two planes, from the plane defined by X
    to the plane defined by Y. In this assignment, X are the coordinates of the
    four corners of the soccer goal while Y are the four corners of the penn logo

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X

    """

    ##### STUDENT CODE START #####
    num_points, _ = X.shape
    A = []
    #Creating the A matrix to perform SVD
    for i in range(num_points):
        ax = np.array([-X[i][0],-X[i][1], -1, 0,0,0, X[i][0]*Y[i][0], X[i][1]*Y[i][0], Y[i][0] ])
        ay = np.array([0,0,0, -X[i][0],-X[i][1], -1, X[i][0]*Y[i][1], X[i][1]*Y[i][1], Y[i][1] ]) 
        A.append(ax)
        A.append(ay)
    
    H = np.linalg.svd(np.array(A))[-1][-1].reshape(3,3) # Performing SVD, taking the last row of V, reshaping into 3x3 matrix and returning it.

    ##### STUDENT CODE END #####

    return H
