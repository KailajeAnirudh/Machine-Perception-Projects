import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    ##### STUDENT CODE START #####
    R_cw = np.linalg.inv(R_wc)
    t_cw = -R_cw @ t_wc
    Pw = np.zeros((pixels.shape[0], 3))
    
    TransformMat = np.hstack((R_cw, t_cw.reshape(3,1)))[:,[0,1,3]] #Remove the z column as the points are assumed to lie on the x-y plane in the world
    H_cw = K @ TransformMat
    H_wc = np.linalg.inv(H_cw) #Compute H matrix from world to camera
    pixel_vec = np.hstack((pixels, np.ones((pixels.shape[0],1)))) # N x 3 matrix of pixel coordinates with 1 appended to each row

    #Compute the world coordinates of the pixels
    Pw = (H_wc @ pixel_vec.T)
    Pw = Pw/Pw[2,:]
    Pw[2] = 0
    Pw = Pw.T
    ##### STUDENT CODE END #####
    return Pw
