#!/usr/bin/env python3
# run_vae_cifar.py
import argparse
import os
import sys
import random
import yaml
import json
import time
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Adjust python path if needed (keeps your repo imports working)
sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import training utilities & model from your module.
# Make sure Models/score_based/vae_cifar.py exports `train`, `show_batch_reconstructions` (or `show_vae_reconstructions`),
# `get_data_loader` (optional), and `VAE32`.
# If your module uses different names, adjust the imports accordingly.
from Models.score_based.vae_cifar import train, show_batch_reconstructions, get_data_loader, VAE32, safe_pil_to_tensor_no_numpy

# -------------------------
# Parse args
# -------------------------
parser = argparse.ArgumentParser(description="Train VAE on CIFAR-10")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument("--device", default=None, help="torch device string, e.g. 'cuda:0' or 'cpu'")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--nepochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--ae_latent_dim", type=int, default=128)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--perceptual_flag", type=eval, default=True)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--weight_perc", type=float, default=0.05)
parser.add_argument("--kl_anneal_epochs", type=int, default=20)
parser.add_argument("--use_amp", action="store_true", help="Use mixed precision if CUDA available")
parser.add_argument("--root", type=str, default="./data")  # dataset root
args = parser.parse_args()


if args.seed is None:
    args.seed = int(np.random.randint(0, 10000))
print("Using seed:", args.seed)

# -------------------------
# Set seeds for reproducibility
# -------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# -------------------------
# Device
# -------------------------
if args.device:
    device = torch.device(args.device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Prepare dataset & dataloaders
# -------------------------
# NOTE: we intentionally do NOT normalize to ImageNet mean/std here because:
#  - VAE decoder outputs are in [0,1] (Sigmoid)
#  - Perceptual loss internally normalizes to ImageNet statistics
# If you *do* normalize here, ensure you denormalize / adapt loss functions accordingly.
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda img: safe_pil_to_tensor_no_numpy(img)),
])
transform_test = transforms.Compose([
    transforms.Lambda(lambda img: safe_pil_to_tensor_no_numpy(img)),  # -> [0,1]
])

# Root for CIFAR dataset (change if needed)
root = "/leonardo/home/userexternal/fgiacoma/repos/Latent_Distilled_Diffusion_4_Cert/data/cifar10"


train_ds = CIFAR10(root=root, train=True, download=False, transform=transform_train)
test_ds = CIFAR10(root=root, train=False, download=False, transform=transform_test)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=(device.type == "cuda"))
val_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))

# -------------------------
# Build model
# -------------------------
# VAE32 signature: VAE32(z_dim=128, base=64)
ae_latent_dims = args.ae_latent_dim
vae_model = VAE32(z_dim=ae_latent_dims).to(device)
print("VAE32 created with latent dim:", ae_latent_dims)

# -------------------------
# Prepare save folder
# -------------------------
if args.modelfolder == "":
    rid = str(np.random.randint(0, 1000))
    model_folder = f"./save/CIFAR_VAE_{ae_latent_dims}/VAE_{rid}/"
else:
    model_folder = args.modelfolder
os.makedirs(model_folder, exist_ok=True)
print("Checkpoints & outputs will be saved to:", model_folder)

# -------------------------
# Train or load
# -------------------------
# If you want to resume from a checkpoint, change this logic accordingly.
print(f"Perceptual loss: {args.perceptual_flag}")
vae = train(
    model=vae_model,
    train_loader=train_loader,
    epochs=args.nepochs,
    lr=args.lr,
    foldername=model_folder,
    device=device,
    perceptual=args.perceptual_flag,
    kl_anneal_epochs=args.kl_anneal_epochs,
    weight_perc=args.weight_perc,
    free_bits=0.5,   # you can set to None if you don't want free-bits
    use_amp=args.use_amp,
)

# -------------------------
# Visualize reconstructions on validation set and save an image
# -------------------------
# The training module exports `show_batch_reconstructions` (name used in the fixed training script).
# If your module exposes `show_vae_reconstructions` instead, either change import above or call that.
show_batch_reconstructions(model=vae, dataloader=val_loader, device=device, n=10, foldername=model_folder)

# -------------------------
# (Optional) save final state explicitly
# -------------------------
final_path = os.path.join(model_folder, "vae_final.pth")
torch.save(vae.state_dict(), final_path)
print("Saved final model to:", final_path)

# (Optional) print final model details
print("Model summary:")
print(vae)
