# Custom Pooling Operations Comparison

This repository contains an implementation and performance comparison of custom pooling operations using NumPy and PyTorch. The goal is to assess the execution speed of a custom-written pooling layer in both frameworks without using the built-in pooling operations, highlighting the efficiency and performance gains achieved with PyTorch.

## Table of Contents

- [Installation](#installation)
- [File Descriptions](#file-descriptions)
- [Running Tests](#running-tests)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributing](#contributing)

## Installation

Before running the scripts, you need to have Python installed on your system. If you don't have Python, install it from [python.org](https://www.python.org/). After setting up Python, you can install the necessary packages using the following command:

    pip install numpy torch

Clone this repository to your local machine to get started:

    git clone https://github.com/yourusername/custom-pooling-comparison.git
    cd custom-pooling-comparison

## File Descriptions

    numpy_pooling.py - This script contains the custom pooling function implemented using NumPy.
    torch_pooling.py - This script contains the custom pooling function implemented using PyTorch, avoiding built-in pooling layers for fairness in comparison.
    performance_test.py - This script is used to compare the performance of the two implementations.

## Running Tests

To run the performance tests, execute the performance_test.py script with the following command:

    python performance_test.py

This script will run the pooling operations multiple times and calculate the average execution time for both the NumPy version and the PyTorch version.

## Results
Tests were conducted on a system equipped with a 3060-12GB GPU. 
The custom pooling operations were applied to images of size 1024x768. 
After running the performance tests, we gathered the following average execution times over 100 iterations:

    TF-based pooling: 1.2378 seconds
    PyTorch-based pooling: 1.002 seconds

