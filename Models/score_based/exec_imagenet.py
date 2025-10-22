import argparse
import torch
import datetime
import time
import json
import yaml
import sys
import os


sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
superdir = os.path.dirname(parent_dir)
import random
#import torch_two_sample 
#from main_mnist import absCSDI
from main_mnist import absCSDI
from utils_mnist import *
import numpy as np
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from Models.score_based.vae_cifar import train, show_vae_reconstructions
from Models.score_based.data_imagenet import SubsetDownsampledImageNet
from Models.score_based.vae_imagenet import VAE16, VAE32, VAE64
from Models.score_based.im64_data import ImageNet64DataModule


parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=-1.0)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--target_dim", type=int, default=2)
parser.add_argument("--eval_length", type=int, default=2)
parser.add_argument("--model_name", type=str, default="IMNET16")
parser.add_argument("--unconditional", default=False)#, action="store_true"
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)
parser.add_argument("--ntrajs", type=int, default=1)
parser.add_argument("--nepochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--scaling_flag", type=eval, default=True)
parser.add_argument("--q", type=float, default=0.9)
parser.add_argument("--load", type=eval, default=False)
parser.add_argument("--rob_flag", type=eval, default=False)
parser.add_argument("--active_flag", type=eval, default=False)
parser.add_argument("--implicit_flag",type=eval, default=True)
parser.add_argument("--map_type", type=str, default = "IMNET16")
parser.add_argument("--gamma", type=float, default=0.0)


#parser.add_argument("--diffsteps", type=int, default=20)
#parser.add_argument("--difflayers", type=int, default=6)
#parser.add_argument("--diffchannels", type=int, default=512)
#parser.add_argument("--diffdim", type=int, default=512)

args = parser.parse_args()
args.seed = np.random.randint(1000)



print(args)

path = os.path.join("Models", "score_based","config",args.config)
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["train"]["epochs"] = args.nepochs
config["train"]["batch_size"] = args.batch_size
config["train"]["lr"] = args.lr
print(args.unconditional)
config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio
config["diffusion"]["input_dim"] = args.target_dim
config["diffusion"]["traj_len"] = args.eval_length
config["diffusion"]["gamma"] = args.gamma
print(config["model"]["is_unconditional"])
#config["diffusion"]["num_steps"] = args.diffsteps
#config["diffusion"]["channels"] = args.diffchannels
#config["diffusion"]["layers"] = args.difflayers
#config["diffusion"]["diffusion_embedding_dim"] = args.diffdim


# code for a latent diffusion model trained on mnist digits data
# This code trains a diffusion model on the MNIST dataset and evaluates it.





# print(json.dumps(config, indent=4))
# if args.modelfolder == "":
#     args.modelfolder = str(np.random.randint(0,500))
#     foldername = f"./save/{args.model_name}/DIFF/ID_UNC{args.modelfolder}/"
#     print('model folder:', foldername)
#     os.makedirs(foldername, exist_ok=True)
#     with open(foldername + "config.json", "w") as f:
#         json.dump(config, f, indent=4)
# else:
#     foldername = f"./save/{args.model_name}/DIFF/ID_UNC{args.modelfolder}/"
#     if args.active_flag:
#         parent_foldername = foldername
#         foldername = parent_foldername+f'FineTune_{args.nepochs}_lr={args.lr}_Q={int(args.q*100)}/'
#         os.makedirs(foldername, exist_ok=True)


print('target dim is: ',args.target_dim)


#aefolder = 401

#ae_foldername = "./save/VAE/VAE_401/"

img_size = 64
bs = 64

ae_foldername = ""

ae_latent_dims: int = 128

print("latent dims: ", ae_latent_dims)

if img_size == 16:
    root="/leonardo_scratch/large/userexternal/fgiacoma/imagenet16"
    vae_model = VAE16(img_size, ae_latent_dims).to(device)
elif img_size == 32:
    root="/leonardo_scratch/large/userexternal/fgiacoma/imagenet32"
    vae_model = VAE32(img_size, ae_latent_dims).to(device)
elif img_size == 64:
    root="/leonardo_scratch/large/userexternal/fgiacoma/imagenet64"
    vae_model = VAE64(img_size, ae_latent_dims).to(device)
else:
    print("invalid img_size")
    exit()
all_labels = list(range(1,1000))
#selected_classes = random.sample(all_labels, 20)

selected_classes = [8, 161, 440, 472, 483, 820, 837, 852, 857, 879]
print("Selected classes:", selected_classes)


# transform = transforms.Compose([
#     # Remove ToTensor()
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])  # if needed
# ])


# train_ds = SubsetDownsampledImageNet(root, 'train', img_size,  selected_classes=selected_classes)
# val_ds   = SubsetDownsampledImageNet(root, 'val',   img_size,  selected_classes=selected_classes)
# train_loader = DataLoader(train_ds, bs, shuffle=True, num_workers=4, pin_memory=True)
# val_loader   = DataLoader(val_ds, bs, shuffle=False, num_workers=4, pin_memory=True)


datamodule = ImageNet64DataModule(
    root_dir=root,
    batch_size=64,
    selected_classes=selected_classes,
)
datamodule.setup()

train_loader=datamodule.train_dataloader(),
val_loader=datamodule.val_dataloader()


# Check if the autoencoder model exists, if not, train it

if ae_foldername == "":
    aefolder = str(np.random.randint(0,500))
    ae_foldername = f"./save/IM{img_size}_VAE/VAE_{aefolder}/"
    print('model folder:', ae_foldername)
    os.makedirs(ae_foldername, exist_ok=True)

    vae = train(model = vae_model, train_loader=train_loader, img_size=img_size, z_dim=ae_latent_dims, epochs= 20, foldername=ae_foldername, perceptual=True)

else:

    #vae.load_state_dict(torch.load(ae_foldername + "vae_model.pth"))
    #print('Autoencoder loaded from:', ae_foldername + "vae_model.pth")
    print("invalid ae_foldername")

with open(os.path.join(ae_foldername, "selected_classes.txt"), "w") as f:
    for c in selected_classes:
        f.write(f"{c}\n")

print("decoded shape: ", vae.decoder(torch.randn(1, ae_latent_dims).to(device)).shape)

show_vae_reconstructions(model=vae, dataloader=val_loader, n=10, device=device, foldername=ae_foldername)


#classifier = train_latent_classifier(vae, train_data, test_data, epochs=30, foldername=ae_foldername)
# Evaluate classifier
#evaluate_latent_classifier(vae, classifier, test_data)


args = get_model_details(args)