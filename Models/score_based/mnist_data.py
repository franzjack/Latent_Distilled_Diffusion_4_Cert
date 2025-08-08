import torch
from torch import nn
from torch.nn import functional as F
from torch import utils
from torch.utils.data import DataLoader,Dataset
from torch import distributions
import torchvision
from torchvision.utils import make_grid


def get_mnist_dataloader(root, batch_size=64):
    transform = torchvision.transforms.ToTensor()

    train_dataloader = DataLoader(
        torchvision.datasets.MNIST(
            root,
            transform=transform,
            train=True,
            download=True
        ),
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        torchvision.datasets.MNIST(
            root,
            transform=transform,
            train=False,
            download=True
        ),
        batch_size=batch_size,
        shuffle=False  
    )

    return train_dataloader, test_dataloader
