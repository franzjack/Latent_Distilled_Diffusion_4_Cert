import yaml
import scipy.io
import os
import math
import pickle
import sys
import diff_stl_models as cm
from utils_diff import *
from tqdm import tqdm
from copy import copy, deepcopy
sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

print(current_dir)
print(parent_dir)

#sys.path.append(os.path.dirname(os.path.abspath('')))

import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *

from boolean_stl import *
import stl
from verification import *

from Models.score_based.main_mnist import Generator, diff_CSDI, absCSDI
from Models.score_based.vae_mnist import FlatVariationalAutoencoder, Decoder, vae_train, LatentClassifier, evaluate_latent_classifier, train_latent_classifier

from Models.score_based.mnist_data import get_mnist_dataloader

# if torch.cuda.is_available() else False
cuda=False
device = 'cuda' if cuda else 'cpu'

print('exec device = ', device)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


train_data, test_data = get_mnist_dataloader(root='./data/', batch_size=64)

print("test_data size: ", len(test_data))


ae_latent_dims: int = 64
train_data, test_data = get_mnist_dataloader(root='./data/', batch_size=64)

vae: nn.Module = FlatVariationalAutoencoder(ae_latent_dims).to(device)

aefolder = 27

ae_foldername = ""
ae_foldername = "./save/VAE_mnist/VAE_111/"


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





conf_path = os.path.join(parent_dir,"Models","score_based","config","base32.yaml")

with open(conf_path, "r") as f:
    config = yaml.safe_load(f)

config["train"]["batch_size"] = 128
config["model"]["test_missing_ratio"] = -1
config["model"]["is_unconditional"] = False
config["diffusion"]["input_dim"] = ae_latent_dims
config["diffusion"]["traj_len"] = ae_latent_dims
config["diffusion"]["gamma"] = 0.3

config_8 = deepcopy(config)
config_4 = deepcopy(config)

config_8["diffusion"]["num_steps"] = 8
config_4["diffusion"]["num_steps"] = 4

config_8["diffusion"]["schedule"] = "custom"
config_4["diffusion"]["schedule"] = "custom"




modelname = "MNIST"
arch = "DIFF"
gen_id = "ID_UNC306"
#gen_id_8 = "ID_UNC111/distill_8"
#gen_id_4 = "ID_UNC111/distill_4"
model_diff = absCSDI(config, device ,target_dim=ae_latent_dims).to(device)

mod_path = os.path.join(parent_dir, 'save', modelname, arch, gen_id, 'fullmodel.pth')
# mod_path_8 = os.path.join(parent_dir, 'save', modelname, arch, gen_id_8, 'student_fullmodel.pth')
# mod_path_4 = os.path.join(parent_dir, 'save', modelname, arch, gen_id_4, 'student_fullmodel.pth')
# path_8 = os.path.join(parent_dir, 'save', modelname, arch, gen_id_8, 'alphas.pkl')
# path_4 = os.path.join(parent_dir, 'save', modelname, arch, gen_id_4, 'alphas.pkl')

# with open(path_8, "rb") as f:
#         alpha_8 = pickle.load(f)

# with open(path_4, "rb") as f:
#         alpha_4 = pickle.load(f)

# model_8 = absCSDI(config_8, device,target_dim=4, teacher_model = None, alphas = alpha_8).to(device)
# model_4 = absCSDI(config_4, device,target_dim=4, teacher_model = None, alphas = alpha_4).to(device)

print('loading model from path: ', mod_path)
model_diff.load_state_dict(torch.load(mod_path, map_location=device))
model = Generator(model_diff)

# model_8.load_state_dict(torch.load(mod_path_8, map_location=device))
# model8 = Generator(model_8)

# model_4.load_state_dict(torch.load(mod_path_4, map_location=device))
# model4 = Generator(model_4)

vae.eval()
model.to(device)
model.eval()

# model8.to(device)
# model8.eval()

# model4.to(device)
# model4.eval()

with tqdm(train_data,
                  leave=True, dynamic_ncols=True, mininterval=0.5) as it:
            batch_list = [(batch_no,(x, _)) for batch_no,(x, _) in enumerate(it, start=1)]

x = batch_list[0][1][0]  # Get the first batch of images
x = x.to(device)
vae.to(device)
print("x device is",x.device)


# Encode input
print(next(vae.parameters()).is_cuda)
with torch.no_grad():
    vae.eval()
    xx = vae.encoder(x)
    #xx.to(model.device)
print("xx device is",xx.device)
xx = xx.unsqueeze(-1)

print("xx shape is",xx.shape)
latent_shape = xx[0,:,:]

latent_shape = latent_shape.unsqueeze(0)  # Add batch dimension

print('latent shape: ', latent_shape.shape)

M_MAX = 2
EPS = 0.001
DELTA_EPS = 0.0005

Zstar = torch.randn_like(latent_shape, device = device)



output = verifier_vanilla(model,Zstar = Zstar,  eps_start = EPS, model_id = "MNIST")

# output_8 = verifier_vanilla(model8, M=M_MAX, eps_start = EPS, Zstar = Zstar, model_id = "MNIST")

# output_4 = verifier_vanilla(model4, M=M_MAX, eps_start = EPS, Zstar = Zstar, model_id = "MNIST")
