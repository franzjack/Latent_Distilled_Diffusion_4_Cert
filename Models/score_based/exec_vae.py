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
from vae import FlatVariationalAutoencoder, Decoder, vae_train, generate_images_grid
from vae import LatentClassifier, train_latent_classifier, ImageClassifier, train_image_classifier, evaluate_latent_classifier ,evaluate_on_reconstructions
from mnist_data import get_mnist_dataloader


import sys
import os


sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
superdir = os.path.dirname(parent_dir)


# Exec code for the variational autoencoder model
# This code trains a variational autoencoder on the MNIST dataset and evaluates it.
# It also trains a classifier on the latent space and evaluates it.
# Additionally, it trains an image classifier on the reconstructed images and evaluates it.
# The code saves the trained models and generates images from the latent space.



ae_latent_dims: int = 2
train_data, test_data = get_mnist_dataloader(root='./data/', batch_size=64)

vae: nn.Module = FlatVariationalAutoencoder(ae_latent_dims).to(device)


ae_foldername = "./save/fVAE/fVAE_137/"


# Check if the autoencoder model exists, if not, train it
if ae_foldername == "":
    aefolder = str(np.random.randint(0,500))
    ae_foldername = f"./save/VAE/VAE_{aefolder}/"
    print('model folder:', ae_foldername)
    os.makedirs(ae_foldername, exist_ok=True)

    vae.train()
    vae: nn.Module = vae_train(vae, train_data, epochs = 20, foldername=ae_foldername)
    vae.eval()

else:
    vae.load_state_dict(torch.load(ae_foldername + "vae_model.pth", map_location=device))
    print('Autoencoder loaded from:', ae_foldername + "vae_model.pth")


vae.eval()


#generate_images_grid(vae, 64, foldername)

classifier = train_latent_classifier(vae, train_data, test_data, epochs=30, foldername=ae_foldername)
# Evaluate classifier
evaluate_latent_classifier(vae, classifier, test_data)

image_classifier = ImageClassifier()
image_classifier = train_image_classifier(image_classifier, train_data, test_data, epochs=30, foldername=ae_foldername)
# Evaluate clean classifier on reconstructed digits
evaluate_on_reconstructions(vae, image_classifier, test_data)







