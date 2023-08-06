import numpy as np

def least_squares_estimation(X1, X2):
  """ YOUR CODE HERE
  """
  # Compute element-wise product between columns of X1 and X2
  A = np.vstack([X1[:, 0]*X2[:, 0], X1[:,0]*X2[:,1], 
                 X1[:,0]*X2[:,2], X1[:,1]*X2[:,0], 
                 X1[:,1]*X2[:,1], X1[:,1]*X2[:,2], 
                 X1[:,2]*X2[:,0], X1[:,2]*X2[:,1], 
                 X1[:,2]*X2[:,2]]).T

  U,S,V = np.linalg.svd(A)

  E = V[-1,:].reshape(3,3).T

  U,S,V = np.linalg.svd(E)
  E = U @ np.diag([1,1,0]) @ V
  """ END YOUR CODE
  """
  return E
