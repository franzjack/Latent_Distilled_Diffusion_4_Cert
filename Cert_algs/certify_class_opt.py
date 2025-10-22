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
from Models.score_based.vae_mnist import FlatVariationalAutoencoder, Decoder, vae_train, ImageClassifier, train_image_classifier

from Models.score_based.mnist_data import get_mnist_dataloader

# if torch.cuda.is_available() else False
cuda=False
device = 'cuda' if cuda else 'cpu'

print('exec device = ', device)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


train_data, test_data = get_mnist_dataloader(root='./data/', batch_size=64)

print("test_data size: ", len(test_data))


class ClassGen(nn.Module):
    def __init__(self, generator, decoder, classifier):
        super(ClassGen, self).__init__()

        self.generator = generator.to(device)
        self.decoder = decoder.to(device)
        self.classifier = classifier.to(device)

    def forward(self, z):
        with torch.no_grad():
            gen_z = self.generator(z)
            signal = self.decoder(gen_z.squeeze(-1))
            pred = self.classifier(signal)
            #pred = torch.argmax(pred, dim=1)

        
        
        #rescaled_signal = ((signal.permute(0,2,1)+1)*self.maxs/2).permute(0,2,1)

        return pred
    
class DecGen(nn.Module):
    def __init__(self, generator, decoder):
        super(DecGen, self).__init__()

        self.generator = generator.to(device)
        self.decoder = decoder.to(device)
        

    def forward(self, z):
        with torch.no_grad():
            gen_z = self.generator(z)
            signal = self.decoder(gen_z.squeeze(-1))


        return signal


ae_latent_dims: int = 4
train_data, test_data = get_mnist_dataloader(root='./data/', batch_size=64)

vae: nn.Module = FlatVariationalAutoencoder(ae_latent_dims).to(device)


aefolder = 137

ae_foldername = ""
ae_foldername = "./save/VAE_mnist/VAE_137/"


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


classifier = ImageClassifier()
classifier.load_state_dict(torch.load(ae_foldername+"image_class.pth", map_location=device))
classifier.eval()





conf_path = os.path.join(parent_dir,"Models","score_based","config","base.yaml")

with open(conf_path, "r") as f:
    config = yaml.safe_load(f)

config["train"]["batch_size"] = 64
config["model"]["test_missing_ratio"] = -1
config["model"]["is_unconditional"] = False
config["diffusion"]["input_dim"] = 4
config["diffusion"]["traj_len"] = 4
config["diffusion"]["gamma"] = 0.3

config_8 = deepcopy(config)
config_4 = deepcopy(config)

config_8["diffusion"]["num_steps"] = 8
config_4["diffusion"]["num_steps"] = 4

config_8["diffusion"]["schedule"] = "custom"
config_4["diffusion"]["schedule"] = "custom"




modelname = "MNIST"
arch = "DIFF"
gen_id = "ID_UNC111"
gen_id_8 = os.path.join(gen_id,"distill_8")
gen_id_4 = os.path.join(gen_id,"distill_4")
model_diff = absCSDI(config, device ,target_dim=4).to(device)

mod_path = os.path.join(parent_dir, 'save', modelname, arch, gen_id, 'fullmodel.pth')
mod_path_8 = os.path.join(parent_dir, 'save', modelname, arch, gen_id_8, 'student_fullmodel.pth')
mod_path_4 = os.path.join(parent_dir, 'save', modelname, arch, gen_id_4, 'student_fullmodel.pth')
path_8 = os.path.join(parent_dir, 'save', modelname, arch, gen_id_8, 'alphas.pkl')
path_4 = os.path.join(parent_dir, 'save', modelname, arch, gen_id_4, 'alphas.pkl')

with open(path_8, "rb") as f:
        alpha_8 = pickle.load(f)

with open(path_4, "rb") as f:
        alpha_4 = pickle.load(f)

model_8 = absCSDI(config_8, device,target_dim=4, teacher_model = None, alphas = alpha_8).to(device)
model_4 = absCSDI(config_4, device,target_dim=4, teacher_model = None, alphas = alpha_4).to(device)

print('loading model from path: ', mod_path)
model_diff.load_state_dict(torch.load(mod_path, map_location=device))
model = Generator(model_diff)

model_8.load_state_dict(torch.load(mod_path_8, map_location=device))
model8 = Generator(model_8)

model_4.load_state_dict(torch.load(mod_path_4, map_location=device))
model4 = Generator(model_4)

vae.eval()
model.to(device)
model.eval()

model8.to(device)
model8.eval()

model4.to(device)
model4.eval()

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

Z0 = torch.randn_like(latent_shape, device = device)

model_dec = DecGen(model, vae.decoder)

Zstar, = optimize_latent_class(z0 = Z0, gen_dec =model_dec, classifier=classifier,target_class=6, l2_reg=0.01)

class_mod = ClassGen(model, vae.decoder, classifier)
class_mod.to(device)
class_mod.eval()

class_mod8 = ClassGen(model8, vae.decoder, classifier)
class_mod8.to(device)
class_mod8.eval()

class_mod4 = ClassGen(model4, vae.decoder, classifier)
class_mod4.to(device)
class_mod4.eval()

pred_base, lb_base, ub_base = verifier_vanilla(class_mod,  eps_start = EPS, Zstar = Zstar, model_id = "MNIST")

pred_8, lb_8, ub_8 = verifier_vanilla(class_mod8,  eps_start = EPS, Zstar = Zstar, model_id = "MNIST")

pred_4, lb_4, ub_4 = verifier_vanilla(class_mod4,  eps_start = EPS, Zstar = Zstar, model_id = "MNIST")


print('Base model prediction: ', nn.functional.softmax(pred_base), ' with bounds: ', nn.functional.softmax(lb_base), nn.functional.softmax(ub_base))

print('8 steps model prediction: ', nn.functional.softmax(pred_8), ' with bounds: ', nn.functional.softmax(lb_8), nn.functional.softmax(ub_8))

print('4 steps model prediction: ', nn.functional.softmax(pred_4), ' with bounds: ', nn.functional.softmax(lb_4), nn.functional.softmax(ub_4))


