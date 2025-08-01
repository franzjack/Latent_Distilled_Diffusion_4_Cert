import argparse
import torch
import datetime
import time
import json
import yaml
import sys
import os
import copy


sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
superdir = os.path.dirname(parent_dir)
#import torch_two_sample 
from main_x import absCSDI
from dataset_cross import *

from utils_norm import *
from utils_distill import train_student, stud_and_teach_eval, plot_compared_many_trajs, plot_compared_trajs
import numpy as np



parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base12.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--model_seed", type=str, default=str(1))
parser.add_argument("--testmissingratio", type=float, default=-1.0)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--target_dim", type=int, default=2)
parser.add_argument("--eval_length", type=int, default=24)
parser.add_argument("--model_name", type=str, default="MAZE")
parser.add_argument("--unconditional", default=False)#, action="store_true"
parser.add_argument("--teacher_folder", type=str, default="")
parser.add_argument("--student_folder", type=str, default="")
parser.add_argument("--modelfolder", type=str, default="491")
parser.add_argument("--nsample", type=int, default=1)
parser.add_argument("--ntrajs", type=int, default=1)
parser.add_argument("--nepochs", type=int, default=5000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--scaling_flag", type=eval, default=True)
parser.add_argument("--q", type=float, default=0.9)
parser.add_argument("--load", type=eval, default=False)
parser.add_argument("--rob_flag", type=eval, default=False)
parser.add_argument("--active_flag", type=eval, default=False)
parser.add_argument("--implicit_flag",type=eval, default=True)
parser.add_argument("--map_type", type=str, default = "obs")
parser.add_argument("--gamma", type=float, default=0.0)

TRAIN = True
SINGLE = False
FIRST_STEP = True
COMPARED = True
next_dist_step = False

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
config_teacher = copy.deepcopy(config)
config_student = copy.deepcopy(config)

original_folder = f"./save/{args.model_name}/DIFF/ID_UNC{args.modelfolder}/"

if FIRST_STEP == True:
    teacher_folder = f"./save/{args.model_name}/DIFF/ID_UNC{args.modelfolder}/"
    config_student["diffusion"]["num_steps"] = int(config_teacher["diffusion"]["num_steps"]/2)
    print('student num_steps:', config_student["diffusion"]["num_steps"])
    next_dist_step = 0
else:
    teacher_folder = f"./save/{args.model_name}/DIFF/ID_UNC{args.modelfolder}/distill_{args.teacher_folder}/"
    config_teacher["diffusion"]["num_steps"] = int(config_teacher["diffusion"]["num_steps"]/(2**next_dist_step))
    config_student["diffusion"]["num_steps"] = int(config_teacher["diffusion"]["num_steps"]/2)
    print('student num_steps:', config_student["diffusion"]["num_steps"])

config_student["diffusion"]["schedule"] = "student"

args.model_seed = str(np.random.randint(0,500))



print('batch size is', config["train"]["batch_size"])


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

print(config_teacher["diffusion"]["schedule"])

original_model = absCSDI(config, args.device,target_dim=args.target_dim).to(args.device)

original_model.load_state_dict(torch.load(original_folder+ "fullmodel.pth"))
if FIRST_STEP == True:
    teacher_model = absCSDI(config_teacher, args.device,target_dim=args.target_dim).to(args.device)
    teacher_model.load_state_dict(torch.load(teacher_folder+ "fullmodel.pth"))
else:
    with open(teacher_folder + "alphas.pkl", "rb") as f:
        alpha_teacher = pickle.load(f)
    config_teacher["diffusion"]["schedule"] = "custom"
    teacher_model = absCSDI(config_teacher, args.device,target_dim=args.target_dim, teacher_model = None, alphas = alpha_teacher).to(args.device)
    teacher_model.load_state_dict(torch.load(teacher_folder+ "student_fullmodel.pth"))

print(json.dumps(config_student, indent=4))

student_model = absCSDI(config_student, args.device,target_dim=args.target_dim, teacher_model= teacher_model).to(args.device)

if args.student_folder == "":
    student_folder = f"./save/{args.model_name}/DIFF/ID_UNC{args.modelfolder}/distill_{args.model_seed}/"
else:
    student_folder = f"./save/{args.model_name}/DIFF/ID_UNC{args.modelfolder}/distill_{args.student_folder}/"
print('model folder:', student_folder)
os.makedirs(student_folder, exist_ok=True)
with open(student_folder + "config.json", "w") as f:
    json.dump(config_student, f, indent=4)

with open(student_folder + "alphas.pkl", "wb") as f:
    pickle.dump(student_model.alpha, f)

print("teacher alphas", teacher_model.alpha_torch)

print("student alphas", student_model.alpha_torch)

if TRAIN:
    st = time.time()
    train_student(
        student_model=student_model,
        teacher_model=teacher_model,
        config =config_student["train"],
        train_loader = train_loader,
        valid_loader=valid_loader,
        foldername=student_folder,
    )
    print('Training time: ', time.time()-st)

else:
    student_model.load_state_dict(torch.load(student_folder+ "student_fullmodel.pth"))



#try:

if COMPARED == True:
    print(student_model.num_steps)
    print('evaluation using the implicit version of the forward process for model')

    stud_and_teach_eval(student_model, original_model, test_loader, nsample=args.nsample, scaler=1, foldername=student_folder, ds_id = 'test',dist_step = 0)

    plot_compared_trajs(opt=args, foldername=student_folder, dataloader=train_loader, nsample=args.nsample, dist_step = 0)

    plot_compared_many_trajs(opt=args, foldername=student_folder, dataloader=train_loader, nsample=args.nsample, dist_step = 0)

    if FIRST_STEP == False:
        stud_and_teach_eval(student_model, original_model, test_loader, nsample=args.nsample, scaler=1, foldername=student_folder, ds_id = 'test',dist_step = next_dist_step)

        plot_compared_trajs(opt=args, foldername=student_folder, dataloader=train_loader, nsample=args.nsample, dist_step = next_dist_step)

        plot_compared_many_trajs(opt=args, foldername=student_folder, dataloader=train_loader, nsample=args.nsample, dist_step = next_dist_step)
        
if SINGLE == True:
    if args.implicit_flag:
        print('evaluation using the implicit version of the forward process for model')
        latent_path = f"./save/{args.model_name}/DIFF/ID_UNC{args.modelfolder}/"+"sampled_test_latent_nsample1" + ".pk"
        with open(latent_path, 'rb') as f:
            latent = pickle.load(f)
        print(type(latent))
        print('latent shape is: ', latent[0].shape)
        stud_evaluate_implicit(student_model, test_loader, nsample=args.nsample, scaler=1, foldername=student_folder, ds_id = 'test',latent=latent[0])
    else:
        evaluate(student_model, test_loader, nsample=args.nsample, scaler=1, foldername=student_folder, ds_id = 'test')



#plot_trajectories(foldername=foldername, nsample=args.nsample)

#plot_rescaled_trajectories(opt=args, foldername=foldername, dataloader=test_loader, nsample=args.nsample, Mred=10)
    plot_rescaled_crossroads2(opt=args, foldername=student_folder, dataloader=train_loader, nsample=args.nsample)

    plot_rescaled_many_trajs(opt=args, foldername=student_folder, dataloader=train_loader, nsample=args.nsample)

# except:
#     print('not able to produce plots')
    

#plot_rescaled_3dline(opt=args, foldername=foldername, dataloader=train_loader, nsample=args.nsample)




print('process terminated for model:', args.model_seed)