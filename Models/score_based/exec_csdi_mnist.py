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
#import torch_two_sample 
#from main_mnist import absCSDI
from main_mnist import absCSDI
from utils_mnist import *
import numpy as np
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
from torch import nn
from Models.score_based.vae_mnist import FlatVariationalAutoencoder, FlatDecoder, vae_train, LatentClassifier, evaluate_latent_classifier, train_latent_classifier
from mnist_data import get_mnist_dataloader


parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=-1.0)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--target_dim", type=int, default=64)
parser.add_argument("--eval_length", type=int, default=64)
parser.add_argument("--model_name", type=str, default="MNIST")
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
parser.add_argument("--map_type", type=str, default = "MNIST")
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


print(json.dumps(config, indent=4))
if args.modelfolder == "":
    args.modelfolder = str(np.random.randint(0,500))
    foldername = f"./save/{args.model_name}/DIFF/ID_UNC{args.modelfolder}/"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)
else:
    foldername = f"./save/{args.model_name}/DIFF/ID_UNC{args.modelfolder}/"
    if args.active_flag:
        parent_foldername = foldername
        foldername = parent_foldername+f'FineTune_{args.nepochs}_lr={args.lr}_Q={int(args.q*100)}/'
        os.makedirs(foldername, exist_ok=True)


print('target dim is: ',args.target_dim)


ae_foldername = ""

ae_latent_dims: int = 64
train_data, test_data = get_mnist_dataloader(root='./data/', batch_size=64)

vae: nn.Module = FlatVariationalAutoencoder(ae_latent_dims).to(device)


#aefolder = 401

ae_foldername = "./save/VAE_mnist/VAE_111/"


# Check if the autoencoder model exists, if not, train it
if ae_foldername == "":
    aefolder = str(np.random.randint(0,500))
    ae_foldername = f"./save/VAE_mnist/VAE_{aefolder}/"
    print('model folder:', ae_foldername)
    os.makedirs(ae_foldername, exist_ok=True)

    vae.train()
    vae: nn.Module = vae_train(vae, train_data, epochs = 50, foldername=ae_foldername)
    vae.eval()

else:
    vae.load_state_dict(torch.load(ae_foldername + "vae_model.pth"))
    print('Autoencoder loaded from:', ae_foldername + "vae_model.pth")

print("decoded shape: ", vae.decoder(torch.randn(1, ae_latent_dims).to(device)).shape)


#classifier = train_latent_classifier(vae, train_data, test_data, epochs=30, foldername=ae_foldername)
# Evaluate classifier
#evaluate_latent_classifier(vae, classifier, test_data)


args = get_model_details(args)


model = absCSDI(config, args.device,target_dim=args.target_dim).to(args.device)

if not args.load:
        st = time.time()
        train_mnist(
            model,
            config["train"],
            train_loader = train_data,
            autoencoder=vae,
            valid_loader=test_data,
            foldername=foldername,
        )
        print('Training time: ', time.time()-st)

else:
    model.load_state_dict(torch.load(foldername+ "fullmodel.pth"))

# Evaluate over the test set
if args.implicit_flag:
    print('evaluation using the implicit version of the forward process for model')
    new_evaluate_implicit_mnist(model, test_data, autoencoder=vae, nsample=args.nsample, scaler=1, foldername=foldername, ds_id = 'test')
else:
    evaluate(model, test_data, nsample=args.nsample, scaler=1, foldername=foldername, ds_id = 'test')



#plot_trajectories(foldername=foldername, nsample=args.nsample)

#plot_rescaled_trajectories(opt=args, foldername=foldername, dataloader=test_loader, nsample=args.nsample, Mred=10)
try:
    plot_results(opt=args, foldername=foldername, autoencoder=vae, nsample=args.nsample)
except:
    print('Error in plotting results, probably due to the shape of the data')
    pass

#plot_results(opt=args, foldername=foldername, autoencoder=vae, nsample=args.nsample)

#plot_rescaled_3dline(opt=args, foldername=foldername, dataloader=train_loader, nsample=args.nsample)




print('process terminated for model:', args.modelfolder)