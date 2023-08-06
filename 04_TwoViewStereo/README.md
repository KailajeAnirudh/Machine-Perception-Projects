# Two-View Stereo Project

This project implements a two-view stereo algorithm to convert multiple 2D viewpoints into a 3D scene reconstruction. The main code is in an interactive Jupyter notebook, `two_view.ipynb`, which imports several functions from `two_view_stereo.py.`

## Results

### Input
![Raw](images.gif)

### Result

![Result](Code/Multiview.gif)

## Requirements

The prerequisite Python libraries are included in `requirements.txt,` which can be installed via `$ pip install -r requirements.txt.` Note that for the K3D library, which is used to visualize the 3D point clouds in the Jupyter Notebook, there are some additional steps after pip installation. Namely, you may run the following lines after installation to explicitly enable the extension:


## Usage

To run the code, open the Jupyter Notebook `two_view.ipynb` and follow the instructions in each cell. The notebook will guide you through the following steps:

- Load and visualize the dataset
- Rectify two views using homography
- Compute disparity map using different metrics (SSD, SAD, ZNCC)
- Compute depth map and point cloud using triangulation
- Post-process and visualize the reconstructed point cloud

The notebook also provides some examples of how to use different view pairs for two-view stereo and how to aggregate the reconstructed point clouds in the world frame.
