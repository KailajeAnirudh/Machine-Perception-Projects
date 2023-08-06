from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation

    H = est_homography(Pw[:,:2], Pc)
    # H = H/(np.linalg.norm(H, ord='fro'))
    Hprime = np.linalg.inv(K) @ H

    # Extracting R and t from Hprime
    # Following slides: Pose from Projective Transformation

    A = np.zeros((3,3))
    A[:,0] = Hprime[:,0]
    A[:,1] = Hprime[:,1]
    A[:,2] = np.cross(Hprime[:,0], Hprime[:,1])
    # A = np.array([Hprime[:,0], Hprime[:,1], np.cross(Hprime[:,0], Hprime[:,1])])

    # Normalizing A
    U, S, V = np.linalg.svd(A)
    R = U @ np.array([[1,0,0],[0,1,0],[0,0,np.linalg.det(U@V.T)]]) @ V
    t = Hprime[:,2]/np.linalg.norm(Hprime[:,0])

    R = np.linalg.inv(R)
    t = R @ -t    

    ##### STUDENT CODE END #####

    return R, t