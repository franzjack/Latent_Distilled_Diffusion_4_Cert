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
import matplotlib.pyplot as plt
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from vae import VariationalAutoencoder, Decoder, vae_train, generate_images_grid
from vae import LatentClassifier, train_latent_classifier, ImageClassifier, train_image_classifier, evaluate_latent_classifier ,evaluate_on_reconstructions
from mnist_data import get_mnist_dataloader


# Exec code for the variational autoencoder model
# This code trains a variational autoencoder on the MNIST dataset and evaluates it.
# It also trains a classifier on the latent space and evaluates it.
# Additionally, it trains an image classifier on the reconstructed images and evaluates it.
# The code saves the trained models and generates images from the latent space.

foldername = f"./save/VAE_10/"

latent_dims: int = 2
train_data, test_data = get_mnist_dataloader(root='./data/', batch_size=64)

vae: nn.Module = VariationalAutoencoder(latent_dims).to(device)


vae.train()
vae: nn.Module = vae_train(vae, train_data, epochs = 20, foldername=foldername)
vae.eval()

for i, (x, y) in enumerate(test_data):
    x: torch.Tensor = x.to(device)
    y: torch.Tensor = y.to(device)
    if i == 0:
        print(f"Shape of x: {x.shape}, y: {y.shape}")
        print(f"Device of x: {x.device}, y: {y.device}")
        print(f"Type of x: {type(x)}, y: {type(y)}")
        print(f"Data type of x: {x.dtype}, y: {y.dtype}")
        print(f"First element of x: {x[0]}, y: {y[0]}")
        print(f"First element of x encoded: {vae.encoder(x[0])}, y: {y[0]}")
        print(f"First element of x decoded: {vae.decoder(vae.encoder(x[0]))[0]}")
        print(f"shape of encoded x: {vae.encoder(x).shape}")
    
        break

#generate_images_grid(vae, 64, foldername)

classifier = train_latent_classifier(vae, train_data, test_data, epochs=30, foldername=foldername)
# Evaluate classifier
evaluate_latent_classifier(vae, classifier, test_data)

image_classifier = ImageClassifier()
image_classifier = train_image_classifier(image_classifier, train_data, test_data, epochs=30, foldername=foldername)
# Evaluate clean classifier on reconstructed digits
evaluate_on_reconstructions(vae, image_classifier, test_data)







