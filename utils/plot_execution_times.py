"""
This file contains the function plot_execution_times for plotting execution time 
profiles of CUDA deep learning model layer functions vs PyTorch implementations.

E6692 Spring 2026

YOU DO NOT NEED TO MODIFY THIS FILE TO COMPLETE THE ASSIGNMENT
"""
import matplotlib.pyplot as plt

INPUT_SIZES = [10, 50, 100, 500, 1000]

def plot_execution_times(time_pytorch_cpu, time_pytorch_gpu, time_cuda, title,
                         marker_size=7, logy=True, figsize=(8, 7)):
    """
    Plots the execution time of pytorch cpu, pytorch gpu, and cuda operations 
    for various input sizes. 
    
    params:
        time_pytorch_cpu, time_pytorch_gpu, time_cuda (list of floats): execution times by input size
        title (string): the title of the plot.
        marker_size=7 (int): size of point on plot
        logy=True (bool): if true the y axis is log scaled, else linear
        figsize=(8, 7): the figure dimensions
    """
    fig = plt.figure(figsize=figsize) # make plot
    axis = fig.add_axes([0,0,1,1])
    if logy:
        axis.set_yscale('log')
    axis.plot(time_pytorch_cpu, color='red', marker='o', ms=marker_size)
    axis.plot(time_pytorch_gpu, color='blue', marker='o', ms=marker_size)
    axis.plot(time_cuda, color='green', marker='o', ms=marker_size)

    plt.xticks([int(i) for i in range(len(INPUT_SIZES))], INPUT_SIZES)
    axis.set_ylabel("Execution Time (ms)")
    axis.set_xlabel("Input Size")

    axis.set_title(title)
    axis.grid()

    axis.legend(["PyTorch CPU", "PyTorch GPU", "CUDA"])
    plt.show()