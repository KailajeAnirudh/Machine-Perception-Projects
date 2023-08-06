# Neural Radiance Fields

This project implements a neural network that can fit a 2D image and a 3D scene using multilayer perceptrons (MLPs) and positional encoding.

## Part 1: Fitting a 2D image

In this part, I trained an MLP to learn a mapping from 2D pixel coordinates to RGB color values, using the Starry Night image by Van Gogh as the target. I used different frequencies of sinusoidal positional encoding to map the input coordinates to a higher dimensional space, and compared the performance of the MLP with and without positional encoding. I used the mean squared error (MSE) as the loss function and the peak signal-to-noise ratio (PSNR) as the quality metric.

### Results
![REsults](./2d.gif)

## Part 2: Fitting a 3D scene

In this part, I trained an MLP to learn a mapping from 3D position and viewing direction to color and density, using multiple 2D views of the same static scene as the input. I used the NeRF architecture from the paper "Neural Radiance Fields for Unconstrained Scene Representation and Rendering". I computed the origins and directions of each camera frame with respect to the world coordinate frame, and sampled points along each ray using stratified sampling. I used the volume rendering equation to compute the color and opacity of each pixel, and used the MSE as the loss function.

### Results
![Results](./Input.gif)

## Dependencies

- PyTorch
- Numpy
- Matplotlib
- PIL

## Usage

- To run part 1, use `python hw5_part1.py`
- To run part 2, use `python hw5_part2.py`
- To plot the camera poses, use `python plot_all_poses.py`
