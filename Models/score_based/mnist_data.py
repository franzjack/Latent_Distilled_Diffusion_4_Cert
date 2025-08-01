import torch
from torch import nn
from torch.nn import functional as F
from torch import utils
from torch.utils.data import DataLoader,Dataset
from torch import distributions
import torchvision
from torchvision.utils import make_grid


def get_mnist_dataloader(root, batch_size=64):

    train_dataloader =  DataLoader(
    torchvision.datasets.MNIST(
        './data/',
        transform=torchvision.transforms.ToTensor(),
        train=True,
        download=True
    ),
    batch_size=128,
    shuffle=True
    )      
    test_dataloader =  DataLoader(
    torchvision.datasets.MNIST(
        './data/',
        transform=torchvision.transforms.ToTensor(),
        train=False,
        download=True
    ),
    batch_size=128,
    shuffle=True
    ) 

    
    return train_dataloader, test_dataloader