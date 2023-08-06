
# Homography Estimation for Virtual Billboarding

![Virtual Billboarding](result.gif)
This was a part of the homework assignments from the CIS5800: Machine Perception Course. The homework file can be found [here](CIS580_2023_HW1_PartA.pdf).

The code is written in Python 3 and uses the following packages:

- numpy
- matplotlib
- opencv-python

The code consists of three files:

`project_logo.py`: This is the main script that runs the logo projection task. It loads a sequence of images from a football match, the corners of the goal in each image, and an image of the Penn Engineering logo. It then calls the functions est_homography and warp_pts to compute the homography between the logo and the goal, and warp the goal points onto the logo points. It then uses inverse warping to copy the logo image onto the video frame, and saves the results as images and a video.

`est_homography.py`: This file contains the function est_homography, which takes two sets of corresponding points (one from the video frame and one from the logo image) and returns the homography matrix that maps the video points onto the logo points. It uses the technique covered in the lectures and Appendix A of the homework pdf.

`warp_pts.py`: This file contains the function warp_pts, which takes a set of sample points (from the video frame), a set of corresponding points (from the logo image), and a homography matrix, and returns the warped positions of the sample points according to the homography. It divides the result by the third term to get homogeneous coordinates.

To run the code, change your current directory to the directory of “Part 1”, and run:

```
python project_logo.py
```


This will generate a resulting image stack in "./part_1_results/” in the same directory.

You can also edit project_logo.py to use your own data, by changing the variables img_dir, logo_img, and corners. Make sure that img_dir contains a sequence of images with consistent names, logo_img is an image of your choice, and corners is a list of four coordinates for each image that define where you want to project your logo.