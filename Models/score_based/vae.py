
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

latent_dims: int = 2

class Encoder(nn.Module):
    def __init__(self, latent_dims: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = torch.flatten(x, start_dim=1)
        x: torch.Tensor = self.linear1(x)
        x: torch.Tensor = F.relu(x)
        x: torch.Tensor = self.linear2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dims: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z: torch.Tensor = self.linear1(z)
        z: torch.Tensor = F.relu(z)
        z: torch.Tensor = self.linear2(z)
        z: torch.Tensor = torch.sigmoid(z)
        z: torch.Tensor = z.reshape((-1, 1, 28, 28))
        return z
    
class FlatDecoder(nn.Module):
    def __init__(self, latent_dims: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z: torch.Tensor = self.linear1(z)
        z: torch.Tensor = F.relu(z)
        z: torch.Tensor = self.linear2(z)
        z: torch.Tensor = torch.sigmoid(z)
        return z
    

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.register_buffer("N_loc", torch.tensor(0.0))
        self.register_buffer("N_scale", torch.tensor(1.0))
        self.kl = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = torch.flatten(x, start_dim=1)
        x: torch.Tensor = self.linear1(x)
        x: torch.Tensor = F.relu(x)
        mu: torch.Tensor =  self.linear2(x)
        sigma: torch.Tensor = torch.exp(self.linear3(x))
        eps = torch.distributions.Normal(self.N_loc, self.N_scale).sample(mu.shape)
        z = mu + sigma * eps
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class Autoencoder(nn.Module):
    def __init__(self, latent_dims: int) -> None:
        super().__init__()
        self.encoder: nn.Module = Encoder(latent_dims)
        self.decoder: nn.Module = Decoder(latent_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z: torch.Tensor = self.encoder(x)
        xx: torch.Tensor = self.decoder(z)
        return xx
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims: int) -> None:
        super().__init__()
        self.encoder: nn.Module = VariationalEncoder(latent_dims)
        self.decoder: nn.Module = Decoder(latent_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z: torch.Tensor = self.encoder(x)
        xx: torch.Tensor = self.decoder(z)
        return xx
    
class FlatVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims: int) -> None:
        super().__init__()
        self.encoder: nn.Module = VariationalEncoder(latent_dims)
        self.decoder: nn.Module = FlatDecoder(latent_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z: torch.Tensor = self.encoder(x)
        xx: torch.Tensor = self.decoder(z)
        return xx
    

def vae_train(autoencoder: nn.Module, data: DataLoader, epochs: int = 20, foldername: str = "") -> nn.Module    :
    tmp_loss: float = 0.0
    opt: torch.optim.Optimizer = torch.optim.Adam(autoencoder.parameters())
    for epoch in trange(epochs):
        for x, _ in data:
            x: torch.Tensor = x.to(device)
            opt.zero_grad()
            x_hat: torch.Tensor = autoencoder(x)
            loss: torch.Tensor = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            tmp_loss: torch.Tensor = ((x - x_hat)**2).sum()
            loss.backward()
            opt.step()
        print(f"Epoca: {(epoch+1)}/{epochs} \t Loss: {tmp_loss:.0f}")
    if foldername != "":
        output_path = foldername + "vae_model.pth"
        torch.save(autoencoder.state_dict(), output_path)
    return autoencoder

def fvae_train(autoencoder: nn.Module, data: DataLoader, epochs: int = 20, foldername: str = "") -> nn.Module    :
    tmp_loss: float = 0.0
    opt: torch.optim.Optimizer = torch.optim.Adam(autoencoder.parameters())
    for epoch in trange(epochs):
        for x, _ in data:
            x: torch.Tensor = x.to(device)
            xf = torch.flatten(x, start_dim=1)
            opt.zero_grad()
            x_hat: torch.Tensor = autoencoder(x)
            loss: torch.Tensor = ((xf - x_hat)**2).sum() + autoencoder.encoder.kl
            tmp_loss: torch.Tensor = ((xf - x_hat)**2).sum()
            loss.backward()
            opt.step()
        print(f"Epoca: {(epoch+1)}/{epochs} \t Loss: {tmp_loss:.0f}")
    if foldername != "":
        output_path = foldername + "vae_model.pth"
        torch.save(autoencoder.state_dict(), output_path)
    return autoencoder

def ae_train(autoencoder: nn.Module, data: DataLoader, epochs: int = 20, foldername: str = "") -> nn.Module    :
    tmp_loss: float = 0.0
    opt: torch.optim.Optimizer = torch.optim.Adam(autoencoder.parameters())
    for epoch in trange(epochs):
        for x, _ in data:
            x: torch.Tensor = x.to(device)
            opt.zero_grad()
            x_hat: torch.Tensor = autoencoder(x)
            loss: torch.Tensor = ((x - x_hat)**2).sum()
            tmp_loss: torch.Tensor = ((x - x_hat)**2).sum()
            loss.backward()
            opt.step()
        print(f"Epoca: {(epoch+1)}/{epochs} \t Loss: {tmp_loss:.0f}")
    if foldername != "":
        output_path = foldername + "ae_model.pth"
        torch.save(autoencoder.state_dict(), output_path)
    return autoencoder


def generate_images_grid(autoencoder: nn.Module, num_samples: int, foldername:str):
    z = torch.randn(num_samples, latent_dims, device=device)
    x = autoencoder.decoder(z)
    fig = plt.figure()
    _ = plt.imshow(make_grid(x).permute(1, 2, 0).cpu().detach().numpy())
    plt.xticks([])
    plt.yticks([])
    plt.show()
    fig.savefig(foldername+f'vae_example.png')



class LatentClassifier(nn.Module):
    def __init__(self, latent_dims: int, num_classes: int = 10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dims, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)


def train_latent_classifier(vae: VariationalAutoencoder,
                            train_loader: DataLoader,
                            test_loader: DataLoader,
                            epochs: int = 10,
                            foldername: str ="") -> LatentClassifier:
    
    classifier = LatentClassifier(latent_dims).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    vae.eval()  # Freeze VAE
    classifier.train()

    print("Training classifier on VAE latent representations...")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                z = vae.encoder(x)

            logits = classifier(z)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f} - Accuracy: {acc:.2f}%")
    if foldername != "":
        output_path = foldername + "latent_class.pth"
        torch.save(classifier.state_dict(), output_path)

    return classifier


def evaluate_latent_classifier(vae: VariationalAutoencoder,
                               classifier: LatentClassifier,
                               test_loader: DataLoader) -> None:
    classifier.eval()
    vae.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            z = vae.encoder(x)
            pred = classifier(z).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"Final Test Accuracy: {100.0 * correct / total:.2f}%")


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
def train_image_classifier(classifier: ImageClassifier,
                           train_loader: DataLoader,
                           test_loader: DataLoader,
                           epochs: int = 10,
                           foldername: str = "") -> ImageClassifier:
    classifier = classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("Training classifier on original MNIST images...")
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = classifier(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f} - Accuracy: {acc:.2f}%")

    if foldername != "":
        output_path = foldername + "image_class.pth"
        torch.save(classifier.state_dict(), output_path)
    return classifier


def evaluate_on_reconstructions(vae: VariationalAutoencoder,
                                classifier: ImageClassifier,
                                test_loader: DataLoader) -> None:
    classifier.eval()
    vae.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x_hat = vae(x)  # decoded image
            logits = classifier(x_hat)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100.0 * correct / total
    print(f"Image Classifier Accuracy on VAE Reconstructions: {acc:.2f}%")
