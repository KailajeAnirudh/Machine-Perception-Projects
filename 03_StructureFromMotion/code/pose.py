import numpy as np

def getRz(theta):
  """ 
  Returns a 3x3 rotation matrix around the z-axis
  Input:
  - theta: rotation angle in radians
  Output:
  - R: 3x3 rotation matrix
  """
  R = np.array([[np.cos(theta),  -np.sin(theta), 0],
                [np.sin(theta),   np.cos(theta), 0],
                [0,                           0, 1]])
  """ 
  """
  return R

def pose_candidates_from_E(E):
  transform_candidates = []
  ##Note: each candidate in the above list should be a dictionary with keys "T", "R"
  """
  REturns a list of 4 possible pose candidates given the essential matrix E.
  Input:
  - E: 3x3 essential matrix
  Output:
  - transform_candidates: a list of 4 possible pose candidates. Each candidate is a dictionary with keys "T", "R"
  YOUR CODE HERE
  """
  U,S,V = np.linalg.svd(E)
  R1 = U @ getRz(np.pi/2).T @ V
  R2 = U @ getRz(-np.pi/2).T @ V
  T1 = U[:,-1]
  T2 = -U[:,-1]
  transform_candidates = [{"T": T1, "R": R1}, {"T": T1, "R": R2}, {"T": T2, "R": R1}, {"T": T2, "R": R2}]
  """ END YOUR CODE
  """
  return transform_candidates