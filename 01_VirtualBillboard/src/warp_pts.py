import numpy as np
from est_homography import est_homography


def warp_pts(X, Y, interior_pts):
    """
    First compute homography from video_pts to logo_pts using X and Y,
    and then use this homography to warp all points inside the soccer goal

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
        interior_pts: Nx2 matrix of points inside goal
    Returns:
        warped_pts: Nx2 matrix containing new coordinates for interior_pts.
        These coordinate describe where a point inside the goal will be warped
        to inside the penn logo. For this assignment, you can keep these new
        coordinates as float numbers.

    """

    # You should Complete est_homography first!
    H = est_homography(X, Y)

    ##### STUDENT CODE START #####

    W = H@np.array([interior_pts.T[0], interior_pts.T[1], np.ones(len(interior_pts))]) #Creating the warped points
    W1 = W/W[2,:] #Creating W[3, :] = 1
    warped_pts = W1[:2].T #returning only the first two rows and then transposing to give (x, y)


    ##### STUDENT CODE END #####

    return warped_pts

