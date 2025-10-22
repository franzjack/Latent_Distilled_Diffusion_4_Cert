import torch
from torch import nn
from torch.nn import functional as F
from torch import utils
from torch import distributions
import torchvision
from torchvision.utils import make_grid
import numpy as np
from tqdm.auto import tqdm
from tqdm.auto import trange
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Models.score_based.vae_mnist import Autoencoder, Decoder, ae_train, generate_images_grid
from mnist_data import get_mnist_dataloader


# exec code for the autoencoder model
# This code trains an autoencoder on the MNIST dataset and evaluates it.

foldername = f"./save/AE_10/"

latent_dims: int = 2
train_data, test_data = get_mnist_dataloader(root='./data/', batch_size=64)

vae: nn.Module = Autoencoder(latent_dims).to(device)

vae.train()
vae: nn.Module = ae_train(vae, train_data, epochs = 20, foldername=foldername)
vae.eval()

for i, (x, y) in enumerate(test_data):
    x: torch.Tensor = x.to(device)
    y: torch.Tensor = y.to(device)
    if i == 0:
        print(f"Shape of x: {x.shape}, y: {y.shape}")
        print(f"Type of x: {type(x)}, y: {type(y)}")
        print(f"Shape of encoded x: {vae.encoder(x).shape}")
        print(f"Shape of encoded x: {vae.encoder(x).unsqueeze(-1).shape}")
        print(f"Shape of encoded x: {vae.encoder(x).unsqueeze(-1).squeeze(-1).shape}")
        print(f"First element of x: {x[0]}, y: {y[0]}")
        print(f"First element of x encoded: {vae.encoder(x[0])}, y: {y[0]}")
        print(f"First element of x decoded: {vae.decoder(vae.encoder(x[0]))[0]}")
        print(f"shape of encoded x: {vae.encoder(x).shape}")
    
        break
#vae.train()