# Structure from Motion (SFM)
## 3D Reconstruction from point-to point correspondence

This project is about 3D reconstruction from two 2D images[2]. The goal is to estimate the essential matrix, the pose and the depth of the scene from two images of a castle.

## Requirements

- Python 3.8 or higher
- numpy
- matplotlib
- opencv-python
- opencv-contrib-python
- jupyterlab



## Description

The project consists of three main parts:

- Estimation of the essential matrix using least-squares and RANSAC methods.
- Pose recovery and 3D reconstruction using SVD decomposition and triangulation.
- Visualization of the epipolar lines, the reprojection errors and the 3D scene.

The project is based on the lecture notes and the "E-matrix" handout. The images are taken with no zoom and no auto-focus, so they share the same matrix of intrinsics. The SIFT descriptors are computed and matched using OpenCV.

## References

- Lecture notes: https://www.cis.upenn.edu/~cis580/spring2023/lectures/lecture8.pdf
- E-matrix handout: https://www.cis.upenn.edu/~cis580/spring2023/hw/hw3/E-matrix.pdf
