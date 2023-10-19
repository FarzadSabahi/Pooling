# Pooling Layer Demonstration

This repository provides a script demonstrating the functionality of max pooling and average pooling operations, commonly used in the field of deep learning, particularly in convolutional neural networks (CNNs).

## Overview

The script defines a `pooling` function, which performs a specified type of pooling operation (max or average) on input data. The input data is typically a 4D array representing a batch of images (with dimensions corresponding to batch size, height, width, and channels). The function then applies pooling across subregions of each image according to the specified pooling size and stride, resulting in a new array of reduced height and width.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed Python 3.x
- You have a working knowledge of Python
- You have installed the numpy library, which can be done via pip:
    pip install numpy

  
## Using the Pooling Script

To use the pooling script, follow these steps:

1. Clone this repository to your local machine.
2. If you haven't already, install the required libraries mentioned in the prerequisites.
3. Navigate to the directory containing the script.
4. Run the script using Python. For example:
    python test_pooling.py

Replace `test_pooling.py` with the actual filename of the script.

The script will execute, and the output will display the results of the pooling operations, demonstrating the changes in the data structure and values.

## Function Description

### `pooling(A_prev, hparameters, mode = "average")`

- **A_prev**: Input data, a numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev).
- **hparameters**: Python dictionary containing "f" and "stride".
- **mode**: The pooling mode ("max" or "average").

The function returns a tuple containing:
- **A**: Output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C).
- **cache**: Cache used in the backward pass of the pooling layer, containing the input and hparameters.



