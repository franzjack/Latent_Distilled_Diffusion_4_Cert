#!/usr/bin/env python
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

# This will download CIFAR-10 to the specified directory
dataset = CIFAR10(
    root='/leonardo/home/userexternal/fgiacoma/repos/Latent_Distilled_Diffusion_4_Cert/data/cifar10',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
print("CIFAR-10 download completed.")
