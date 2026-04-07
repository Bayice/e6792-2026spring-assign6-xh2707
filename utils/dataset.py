"""
This file contains the functions for loading and displaying the MNIST digit
dataset.

E6692 Spring 2026

YOU DO NOT NEED TO MODIFY THIS FILE TO COMPLETE THE ASSIGNMENT
"""
import torch
import torchvision
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
BATCH_SIZE = 4
DATA_PATH = './data'
NUM_WORKERS = 2
NUM_TRAINING_IMAGES = 1000
NUM_VALIDATION_IMAGES = 100

def load_mnist_dataset():
    """
    Loads the CIFAR10 dataset. This function downloads the images and 
    returns training and validation DataLoader objects.
    
    See here for details: 
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """
    transform = transforms.Compose([transforms.ToTensor()]
                                  )
    
    train = torchvision.datasets.MNIST(root=DATA_PATH, train=True,
                                         download=True, transform=transform)

    validation = torchvision.datasets.MNIST(root=DATA_PATH, train=False,
                                              download=True, transform=transform)
    
    training_indices = np.arange(0, NUM_TRAINING_IMAGES)
    
    validation_indices = np.arange(0, NUM_VALIDATION_IMAGES)
    
    train_loader = Subset(train, training_indices)
    validation_loader = Subset(validation, validation_indices)
    
    training_set = DataLoader(train_loader, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    
    validation_set = DataLoader(validation_loader, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=NUM_WORKERS)
    
    return training_set, validation_set

    
def display_image(image, title=None):
    """
    Display BATCH_SIZE images.
    """
    numpy_image = image.numpy()
    plt.imshow(np.transpose(numpy_image, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
    
    
