import numpy as np

def reconstruct3D(transform_candidates, calibrated_1, calibrated_2):
  """This functions selects (T,R) among the 4 candidates transform_candidates
  such that all triangulated points are in front of both cameras.
  """

  best_num_front = -1
  best_candidate = None
  best_lambdas = None
  for candidate in transform_candidates:
    R = candidate['R']
    T = candidate['T']

    lambdas = np.zeros((2, calibrated_1.shape[0]))
    """ YOUR CODE HERE
    """
    # loop over each row in the data
    for i in range(calibrated_2.shape[0]):
        # normalize the vectors
        calibrated_1_norm = calibrated_1[i,:] #/ np.linalg.norm(calibrated_1[i,:])
        calibrated_2_norm = calibrated_2[i,:] #/ np.linalg.norm(calibrated_2[i,:])
        
        # create the matrix A
        A = np.column_stack((calibrated_2_norm, -R @ calibrated_1_norm ))
        
        # calculate the pseudoinverse of A and compute lambda
        lambdas[:,i] = T @ np.linalg.pinv(A) # @ (T )#/ np.linalg.norm(T))

    """ END YOUR CODE
    """
    num_front = np.sum(np.logical_and(lambdas[0]>0, lambdas[1]>0))

    if num_front > best_num_front:
      best_num_front = num_front
      best_candidate = candidate
      best_lambdas = lambdas
      print("best", num_front, best_lambdas[0].shape)
    else:
      print("not best", num_front)


  P1 = best_lambdas[1].reshape(-1, 1) * calibrated_1
  P2 = best_lambdas[0].reshape(-1, 1) * calibrated_2
  T = best_candidate['T']
  R = best_candidate['R']
  return P1, P2, T, R