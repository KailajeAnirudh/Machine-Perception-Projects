import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

     ##### STUDENT CODE START #####
    #Using Grunert's algorithm to solve the P3P problem (Reference: https://haralick-org.torahcode.us/journals/three_point_perspective.pdf)

    #Converting the pixels into world cordinates upto a scale
    j1, j2, j3, _ = (np.linalg.inv(K) @ np.hstack((Pc, np.ones((Pc.shape[0], 1)))).T).T

    #Getting unit vectors for j1, j2, and j3
    j1 = j1 / np.linalg.norm(j1)
    j2 = j2 / np.linalg.norm(j2)
    j3 = j3 / np.linalg.norm(j3)

    # print("j1, j2, j3: ", j1, j2, j3)

    #Getting the cosine of angles between the vectors
    c_alpha = np.dot(j2, j3)
    c_beta = np.dot(j1, j3)
    c_gamma = np.dot(j1, j2)

    #Distance between the points
    a = np.linalg.norm(Pw[1] - Pw[2])
    b = np.linalg.norm(Pw[0] - Pw[2])
    c = np.linalg.norm(Pw[0] - Pw[1])

    #Calculating square of the distance between the points
    a2 = a**2
    b2 = b**2
    c2 = c**2

    #Calcultating the quartic coefficients
    #Calcultating the quartic coefficients
    #Calcultating the quartic coefficients
    A4 = (((a*a - c*c)/b**2)-1)**2 - 4*c*c*(c_alpha**2)/(b*b)
    A3 = 4*( (((a*a - c*c)/b**2))*(1 - ((a*a - c*c)/b**2))*c_beta - (1 - (a**2+c**2)/b**2)*c_alpha*c_gamma + 2*c*c*c_alpha*c_alpha*c_beta/b**2 )
    A2 = 2*( (((a*a - c*c)/b**2))**2 - 1 + 2*(((a*a - c*c)/b**2)*((a*a - c*c)/b**2))*c_beta*c_beta + 2*(b**2 - c**2)*c_alpha*c_alpha/b**2 - 4*((a*a + c*c)/b**2)*c_alpha*c_beta*c_gamma + 2*c_gamma*c_gamma*(b*b - a*a)/b**2 )
    A1 = 4*(   -(((a*a - c*c)/b**2))*(1+((a*a - c*c)/b**2))*c_beta + 2*a*a*c_gamma*c_gamma*c_beta/b**2 - (1 - (a*a + c*c)/b**2)*c_alpha*c_gamma)
    A0 =  (1+((a*a - c*c)/b**2))**2 - 4*a*a*c_gamma**2/b**2

    #Calculating the roots of the quartic equation
    roots = np.roots([A4, A3, A2, A1, A0])

    #It's possible that the roots are complex, so we need to take only the real part
    #Since u and v must be positive, we need to take only the positive roots
    roots = roots[np.logical_and((abs(roots.imag) < 1e-4),roots.real > 0)].real

    #Calculating the values of u and v
    u_values = ((-1 + ((a2-c2)/b2))*roots**2 - 2*((a2-c2)/b2)*c_beta*roots + 1 + ((a2-c2)/b2))/(2*(c_gamma - c_alpha*roots))

    vu_pairs = np.vstack((roots, u_values)).T
    vu_pairs = vu_pairs[np.all(vu_pairs > 0, axis=1)]
    # print("vu_pairs: ", vu_pairs)
    
    #Caluculating values of s1
    s1_values = np.array([np.sqrt(a2/(u**2 + v**2 - 2*u*v*c_alpha)) for v,u in vu_pairs])
    #Calculating disytances from s1, u, and v
    s1s2s3_sets = np.vstack((s1_values, s1_values*vu_pairs[:, 1], s1_values*vu_pairs[:, 0])).T

    Rcandidates = np.zeros((len(s1s2s3_sets), 3, 3))
    tcandidates = np.zeros((len(s1s2s3_sets), 3))
    Candidate_errors = np.zeros((len(s1s2s3_sets)))

    for i, scales in enumerate(s1s2s3_sets):
        s1, s2, s3 = scales
        #Calcualting the coordinates of the points in the camera frame
        Pc_3d = np.array([s1*j1, s2*j2, s3*j3])
       
        #Calculating the candidate rotation matrix and translation vector
        Rcandidates[i], tcandidates[i] = Procrustes(Pc_3d, Pw[0:3])

        #Calculating the error actual pixels and project pixel coordinates
        Calcord = Rcandidates[i] @ (K[0][0]*np.linalg.inv(K) @ np.array([Pc[-1,0], Pc[-1,1], 1]).T) + tcandidates[i]
        Candidate_errors[i] = np.linalg.norm(Pw[-1,:] - Calcord)
    
    #Selecting the candidate with the least error
    best_candidate = np.argmin(Candidate_errors)
    R = Rcandidates[best_candidate]
    t = tcandidates[best_candidate]
    
    # Invoke Procrustes function to find R, t
    # You may need to select the R and t that could transoform all 4 points correctly. 
    # R,t = Procrustes(Pc_3d, Pw[1:4])
    ##### STUDENT CODE END #####

    return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    ##### STUDENT CODE START #####
    #Calculating the mean of the points
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)

    #Calculating the centered points
    X_centered = (X - X_mean).T
    Y_centered = (Y - Y_mean).T

    #Calculating the covariance matrix
    cov_mat = Y_centered@X_centered.T

    #Calculating the SVD of the covariance matrix
    U, S, V = np.linalg.svd(cov_mat)

    #Calculating the rotation matrix

    R = U@np.array([[1, 0,                      0], 
                    [0, 1,                      0], 
                    [0, 0, np.linalg.det(V.T@U.T)]]) @ V

    #Calculating the translation vector
    t = Y_mean - R@X_mean



    ##### STUDENT CODE END #####

    return R, t
