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
from main_x import absCSDI
from dataset_cross import *

from utils_norm import *
import numpy as np



parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=-1.0)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--target_dim", type=int, default=2)
parser.add_argument("--eval_length", type=int, default=12)
parser.add_argument("--model_name", type=str, default="OBS")
parser.add_argument("--unconditional", default=False)#, action="store_true"
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)
parser.add_argument("--ntrajs", type=int, default=1)
parser.add_argument("--nepochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--scaling_flag", type=eval, default=True)
parser.add_argument("--q", type=float, default=0.9)
parser.add_argument("--load", type=eval, default=False)
parser.add_argument("--rob_flag", type=eval, default=False)
parser.add_argument("--active_flag", type=eval, default=False)
parser.add_argument("--implicit_flag",type=eval, default=True)
parser.add_argument("--map_type", type=str, default = "obs")
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
train_loader, valid_loader, test_loader = get_dataloader(
    model_name=args.model_name,
    eval_length=args.eval_length,
    target_dim=args.target_dim,
    datafolder=args.map_type,
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
    scaling_flag=args.scaling_flag,
    path=superdir
)

print('train_set dim is: ', train_loader.dataset.target_dim)
print('train_set shape is: ', train_loader.dataset.observed_values.shape)
print('test_set shape is: ', test_loader.dataset.observed_values.shape)
print("train maxes are ", train_loader.dataset.max)
print("train mins are ", train_loader.dataset.min)




args = get_model_details(args)


model = absCSDI(config, args.device,target_dim=args.target_dim).to(args.device)

if not args.load:
        st = time.time()
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
        print('Training time: ', time.time()-st)

else:
    model.load_state_dict(torch.load(foldername+ "fullmodel.pth"))

# Evaluate over the test set
if args.implicit_flag:
    print('evaluation using the implicit version of the forward process for model')
    new_evaluate_implicit(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername, ds_id = 'test')
else:
    evaluate(model, train_loader, nsample=args.nsample, scaler=1, foldername=foldername, ds_id = 'test')



#plot_trajectories(foldername=foldername, nsample=args.nsample)

#plot_rescaled_trajectories(opt=args, foldername=foldername, dataloader=test_loader, nsample=args.nsample, Mred=10)
plot_rescaled_crossroads2(opt=args, foldername=foldername, dataloader=train_loader, nsample=args.nsample)

plot_rescaled_many_trajs(opt=args, foldername=foldername, dataloader=train_loader, nsample=args.nsample)

#plot_rescaled_3dline(opt=args, foldername=foldername, dataloader=train_loader, nsample=args.nsample)




print('process terminated for model:', args.modelfolder)








#plot_histograms(opt=args, foldername=foldername, dataloader=train_loader, nsample=args.nsample)

# Compute Wasserstein distance over test set
#wd = compute_wass_distance(opt=args,model=model, dataloader=train_loader, nsample=args.nsample, scaler=1, foldername=foldername)
#res_wd = compute_rescaled_wass_distance(opt=args,model=model, dataloader=train_loader, nsample=args.nsample, scaler=1, foldername=foldername)

# Statistical tests
#statistical_test(opt=args,model=model, dataloader=train_loader, nsample=args.nsample, scaler=1, foldername=foldername)



# Evaluate over the validation set
#if not args.load:
 #   evaluate(model, valid_loader, nsample=args.nsample, scaler=1, foldername=foldername, ds_id = 'valid')
#plot_rescaled_trajectories(opt=args, foldername=foldername, dataloader=train_loader, nsample=args.nsample, Mred=10)

#plot_histograms(opt=args, foldername=foldername, dataloader=train_loader, nsample=args.nsample)    
    

    