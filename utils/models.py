"""
This file contains the PyTorchClassifier class that defines a PyTorch implemented CNN
Classifier and the CUDAClassifier class that defines a CUDA version of PyTorchClassifier.

E6692 Spring 2026
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .dataset import CLASSES
from .context import Context, GPUKernels

BLOCK_SIZE = 32

class PyTorchClassifier(nn.Module):
    """
    Basic CNN Classifier that inherits 
    from the nn.Module class. Convolutional layers
    are restricted to only 1 filter because of our 2D assumption
    for the CUDA implementation of conv2d().
    
    For more information about how multiple filters are handled 
    in conv2d visit this link:
    
    https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
    
    DO NOT MODIFY THIS CLASS
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 5, padding="same", bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(1, 1, 5, padding="same", bias=False)
        self.fc1 = nn.Linear(49, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(CLASSES))

    def forward(self, x):
        """
        The forward pass of the model. 
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
    
class CUDAClassifier(nn.Module):
    """
    The CUDA implementation of PyTorchClassifier. This class
    replicates the functionality of the network defined above, 
    but insead of using PyTorch layer functions, it uses the 
    CUDA kernel functions we defined in Part 1. 
    """
    def __init__(self, kernel_path):
        """
        Initialize the GPU context, read the kernel
        functions in to the source module
        
        param:
            kernel_path (string): path to the CUDA kernel function source file
        """
        super().__init__()
        context = Context(BLOCK_SIZE)
        source_module = context.getSourceModule(kernel_path)
        self.kernels = GPUKernels(context, source_module)
        
        self.state_dict = None # To be defined in load_state_dict()
        
        
    def load_state_dict(self, state_dict):
        """
        Load the saved weights from PyTorchClassifier to
        CUDAClassifier. This function initializes the self.state_dict
        attribute and converts the PyTorch weights from torch.tensor to np.array.
        
        Use weight.cpu().numpy() to:
            A. move the weight from GPU to CPU memory
            B. convert from torch.tensor to np.array
        
        param:
            state_dict (string): path to the saved PyTorchClassifier weights
        """
        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################
    
#         raise Exception('utils.models.GPUKernels.load_state_dict() not implemented!') # delete me
        self.state_dict = {}

        for key, weight in state_dict.items():
            self.state_dict[key] = weight.cpu().numpy()
        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################


            
    def forward(self, x):
        """
        The forward pass of the CUDA classifier model. This function
        should replicate the operations of PyTorchClassifier.forward() 
        but use the CUDA functions defined in self.kernels instead of 
        PyTorch layer functions. The trained weights (convolutional kerneld, 
        dense layer weights and biases) are accessed from self.state_dict
        after they are loaded with self.load_state_dict().
        
        param:
            x (np.array): the input image
        
        returns:
            x (np.array): the output prediction
        """     
        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################
    
#         raise Exception('utils.models.GPUKernels.forward() not implemented!') # delete me
        
        # HINT: Use the functions of self.kernels to call the layers.
        if len(x.shape) == 3:
            x = x[:, :, 0]

        x = x.astype(np.float32)

        # conv1
        conv1_weight = self.state_dict['conv1.weight'][0, 0]
        x = self.kernels.conv2d(x, conv1_weight)

        # relu
        x = self.kernels.relu(x)

        # pool
        x = self.kernels.MaxPool2d(x, kernel_size=2)

        # conv2
        conv2_weight = self.state_dict['conv2.weight'][0, 0]
        x = self.kernels.conv2d(x, conv2_weight)

        # relu
        x = self.kernels.relu(x)

        # pool
        x = self.kernels.MaxPool2d(x, kernel_size=2)

        # flatten
        x = self.kernels.flatten(x)

        # fc1
        fc1_weight = self.state_dict['fc1.weight']
        fc1_bias = self.state_dict['fc1.bias']
        x = self.kernels.linear(x, fc1_weight, fc1_bias)

        # relu
        x = self.kernels.relu(x)
        x = self.kernels.flatten(x)

        # fc2
        fc2_weight = self.state_dict['fc2.weight']
        fc2_bias = self.state_dict['fc2.bias']
        x = self.kernels.linear(x, fc2_weight, fc2_bias)

        # relu
        x = self.kernels.relu(x)
        x = self.kernels.flatten(x)

        # fc3
        fc3_weight = self.state_dict['fc3.weight']
        fc3_bias = self.state_dict['fc3.bias']
        x = self.kernels.linear(x, fc3_weight, fc3_bias)

        x = self.kernels.flatten(x)
        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        return x
        
        