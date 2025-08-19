import yaml
import scipy.io
import os
import math
import pickle
import sys
import diff_stl_models as cm
from utils_diff import *
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

from Models.score_based.new_main import Generator, diff_CSDI, absCSDI

# if torch.cuda.is_available() else False
cuda=False
device = 'cuda' if cuda else 'cpu'

print('exec device = ', device)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


case_path = os.path.join(parent_dir,"Cert_algs","Cases","diff","half_cross.yaml")

with open(case_path, "r") as f:
    case_data = yaml.safe_load(f)


modelname = case_data['modelname']
dataname = case_data['dataname']

arch = 'DIFF'
gen_id = case_data['gen_id']

traj_len = case_data['traj_len']
xdim = case_data['xdim']
latent_dim = case_data['latent_dim']

if dataname == 'obs':
    from Models.score_based.dataset_cross import get_dataloader 
else:
    from Models.score_based.dataset_short import get_dataloader 

train_loader, valid_loader, test_loader = get_dataloader(
    model_name=modelname,
    eval_length=traj_len,
    target_dim=xdim,
    datafolder=dataname,
    seed=np.random.randint(9),
    nfold=0,
    batch_size=64,
    missing_ratio=-1,
    scaling_flag=True,
    path=parent_dir
)


MAXES = train_loader.dataset.max
MINS = train_loader.dataset.min

IND = 0

class CondGenerator(nn.Module):
    def __init__(self, generator, obs_data):
        super(CondGenerator, self).__init__()

        self.generator = generator.to(device)
        self.data = obs_data.to(device)
        self.maxs = torch.tensor(MAXES).to(device) 

    def forward(self, noise):

        signal = self.generator(self.data, noise).to(device)
        rescaled_signal = self.mins + (signal+1)*(self.maxs-self.mins)/2
        
        #rescaled_signal = ((signal.permute(0,2,1)+1)*self.maxs/2).permute(0,2,1)

        return rescaled_signal



CondGenerator_w_STL = getattr(cm, 'condGen_w_BoolSTL_'+dataname)
CondGenerator_w_QuantSTL = getattr(cm, 'condGen_w_QuantSTL_'+dataname)

print('traj_len is ', traj_len)


#conf_path = os.path.join(os.path.abspath(''),"Models/score_based/config/base.yaml")
conf_path = os.path.join(parent_dir,"Models","score_based","config","base.yaml")

with open(conf_path, "r") as f:
    config = yaml.safe_load(f)


config["train"]["batch_size"] = 64
config["model"]["test_missing_ratio"] = -1
config["model"]["is_unconditional"] = False
config["diffusion"]["input_dim"] = xdim
config["diffusion"]["traj_len"] = traj_len

## NOTA : fix for test dimensions


model_diff = absCSDI(config, device ,target_dim=xdim).to(device)

#mod_path = os.path.join(os.path.abspath(''), 'save', modelname, arch, gen_id, 'fullmodel.pth')
mod_path = os.path.join(parent_dir, 'save', modelname, arch, gen_id, 'fullmodel.pth')
print('loading model from path: ', mod_path)
model_diff.load_state_dict(torch.load(mod_path, map_location=device))
model = Generator(model_diff)

model.eval()

from tqdm import tqdm
print('test dimension is', test_loader.dataset.eval_length)
with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
  batch_list = [(batch_no,test_batch) for batch_no,test_batch in enumerate(it, start=1)]

obs_data = model_diff.process_data(batch_list[0][1])[0,:,:]

obs_data = obs_data.unsqueeze(0)

print('obs_data shape is: ', obs_data.shape)

latent_shape = obs_data[:,:,1:]

cmodel = CondGenerator(model, obs_data)
cmodel.eval()

bmodel = CondGenerator_w_STL(model, obs_data,MAXES,MINS)
bmodel.eval()

qmodel = CondGenerator_w_QuantSTL(model, obs_data,MAXES,MINS)
qmodel.eval()

M_MAX = 2
EPS = 0.001
DELTA_EPS = 0.0005
NSAMPLES = 200
WEIGHT = 0.4
CERTIFICATION = True
GEN_TRAJ = True
GUIDANCE = False
VANILLA_SAT = False
#for the generation purposes
ex_id = 'test'

HOM = True
HET = False

if HOM:
    cert_type = 'HOM'
else:
    cert_type = 'HET'

hom_results_path = os.path.join(parent_dir, 'Cert_algs',"results","DIFF",dataname,cert_type, "diff_small_eps_"+dataname+"_Hom_results.pkl")
het_results_path = os.path.join(parent_dir, 'Cert_algs',"results","DIFF",dataname,cert_type, "diff_small_eps_"+dataname+"_Het_results.pkl")
hom_list_path = os.path.join(parent_dir, 'Cert_algs',"results","DIFF",dataname,cert_type, "diff_small_eps_"+dataname+"_Hom_list.pkl")
het_list_path = os.path.join(parent_dir, 'Cert_algs',"results","DIFF",dataname,cert_type, "diff_small_eps_"+dataname+"_Het_list.pkl")

if CERTIFICATION:
    

    if HOM:
        print('-------- HOMEGENEOUS INCREMENTS----------')
        #ball_list = verifier(bmodel, qmodel, M=10, epsilon = 0.1, latent_dim = latent_dim)
        output = verifier_vanilla(bmodel, M=M_MAX, eps_start = EPS, latent_shape = latent_shape, model_id = dataname)
        hom_ball_list = verifier_increment2(bmodel, qmodel, M=M_MAX, eps_start = EPS, delta_eps = DELTA_EPS, latent_shape = latent_shape,model_id = dataname)
        hom_balls = {"hom_ball_list": hom_ball_list}
        with open(hom_list_path, "wb") as f:
            pickle.dump(hom_balls, f)

        Mhom = len(hom_ball_list)
        print('nb of balls = ', Mhom)


        #hom_Z, hom_robs = empirical_satisfaction(hom_ball_list, qmodel, latent_dim, nsamples = NSAMPLES, model_name = 'obstacles homogeneous')

        #hom_log_prob = compute_log_prob(hom_ball_list, latent_dim)

        #print('logprob = ', hom_log_prob)

        #hom_results = {"hom_ball_list": hom_ball_list, "hom_log_prob": hom_log_prob, 'hom_trajs': hom_Z, 'M_MAX': M_MAX, 'EPS': EPS, 'DELTA_EPS': DELTA_EPS, 'NSAMPLES': NSAMPLES,}
        
        #with open(hom_results_path, "wb") as f:
        #    pickle.dump(hom_results, f)
    
    if HET:
        print('-------- HETEROGENEOUS INCREMENTS----------')

        het_ball_list = verifier_heterog_increment2(bmodel, qmodel, weight = WEIGHT, M=M_MAX, epsilon = EPS, delta_eps = DELTA_EPS, latent_shape = latent_shape,model_id = dataname)

        Mhet = len(het_ball_list)
        print('nb of balls = ', Mhet)


        #heter_Z, heter_robs = empirical_satisfaction(heter_ball_list, qmodel, latent_dim, nsamples = NSAMPLES, model_name = 'obstacles heterogeneous')


        #heter_log_prob = compute_log_prob(heter_ball_list, latent_dim)

        #print('logprob = ', heter_log_prob)


        #het_results = {"heter_ball_list": heter_ball_list, "heter_log_prob": heter_log_prob, 'heter_trajs': heter_Z, 'WEIGHT': WEIGHT, 'M_MAX': M_MAX, 'EPS': EPS, 'DELTA_EPS': DELTA_EPS, 'NSAMPLES': NSAMPLES }
        
        #with open(het_results_path, "wb") as f:
        #    pickle.dump(het_results, f)

    


if GEN_TRAJ:

    latent_dim = latent_dim -1

    mean = torch.zeros(latent_dim, device=device)  
    covariance_matrix = torch.eye(latent_dim, device=device)  
    
    mvn = MultivariateNormal(mean, covariance_matrix)

    
    vanilla_trajs,vanilla_z = diff_vanilla_samples(model, obs_data, NSAMPLES, latent_shape, qmodel.mins, qmodel.maxs)

    vanilla_log_lkh = mvn.log_prob(vanilla_z).sum()

    try:
        print("----> vanilla_log_lkh = ", vanilla_log_lkh)
    except:
        print("vanilla not done in this run, are you sure everything is correct?")

    baseline_results = {'vanilla_trajs': vanilla_trajs, 'vanilla_log_lkh' : vanilla_log_lkh,}

    if VANILLA_SAT:
        ex_id = ex_id + "_VS_"

        vanilla_sat_z = diff_get_vanilla_sat_samples(qmodel, NSAMPLES, latent_shape)


        vanilla_sat_log_lkh = mvn.log_prob(vanilla_sat_z).sum()

        baseline_results['vanilla_sat_z'] = vanilla_sat_z
        baseline_results['vanilla_sat_log_lkh'] = vanilla_sat_log_lkh
        baseline_path = os.path.join(parent_dir, 'Cert_algs',"results","DIFF",dataname,'EXP',"diff_"+dataname+"_"+ex_id+"_baseline_results.pkl")
        with open(baseline_path, "wb") as f:
            pickle.dump(baseline_results, f)



    if GUIDANCE:

        ex_id = ex_id +"_GUID_"
        guidance_trajs, guidance_z = guidance_samples(qmodel, latent_shape, NSAMPLES)
        guidance_log_lkh = mvn.log_prob(guidance_z).sum()

        baseline_results['guidance_z'] = guidance_z
        baseline_results['guidance_trajs'] = guidance_trajs
        baseline_results['guidance_log_lkh'] = guidance_log_lkh
        baseline_path = os.path.join(parent_dir, 'Cert_algs',"results","DIFF",dataname,'EXP',"diff_"+dataname+"_"+ex_id+"_baseline_results.pkl")
        with open(baseline_path, "wb") as f:
            pickle.dump(baseline_results, f)


    if HOM:
        ex_id = ex_id + cert_type


        with open(hom_list_path, 'rb') as file:
            hom_ball_list_dict = pickle.load(file)
        hom_ball_list= hom_ball_list_dict["hom_ball_list"]
        hom_Z, hom_robs = empirical_satisfaction(hom_ball_list, qmodel, latent_dim,latent_shape, nsamples = NSAMPLES, model_name = dataname)

        hom_log_prob = compute_log_prob(hom_ball_list, latent_dim)

        print('logprob = ', hom_log_prob)

        hom_results = {"hom_ball_list": hom_ball_list, "hom_log_prob": hom_log_prob, 'hom_trajs': hom_Z, 'M_MAX': M_MAX, 'EPS': EPS, 'DELTA_EPS': DELTA_EPS, 'NSAMPLES': NSAMPLES,}
        
        with open(hom_results_path, "wb") as f:
            pickle.dump(hom_results, f)

       

        with open(hom_results_path, 'rb') as file:
            hom_results = pickle.load(file)

        

        hom_z = hom_results['hom_trajs']
        hom_log_lkh = mvn.log_prob(hom_z).sum()

        hom_trajs = get_trajs(model, obs_data, hom_z, qmodel.mins, qmodel.maxs)

        baseline_results['hom_z'] = hom_z
        baseline_results['hom_trajs'] = hom_trajs
        baseline_results['hom_log_lkh'] = hom_log_lkh

        baseline_path = os.path.join(parent_dir, 'Cert_algs',"results","DIFF",dataname,'EXP',"diff_"+dataname+"_"+ex_id+"_baseline_results.pkl")
        with open(baseline_path, "wb") as f:
            pickle.dump(baseline_results, f)
    
    if HET:

        ex_id = ex_id + cert_type

        het_Z, heter_robs = empirical_satisfaction(het_ball_list, qmodel, latent_dim,latent_shape, nsamples = NSAMPLES, model_name = dataname+'_heterogeneous')


        heter_log_prob = compute_log_prob(het_ball_list, latent_dim)

        print('logprob = ', heter_log_prob)


        het_results = {"heter_ball_list": het_ball_list, "heter_log_prob": heter_log_prob, 'heter_trajs': het_Z, 'WEIGHT': WEIGHT, 'M_MAX': M_MAX, 'EPS': EPS, 'DELTA_EPS': DELTA_EPS, 'NSAMPLES': NSAMPLES }
        
        with open(het_results_path, "wb") as f:
            pickle.dump(het_results, f)

        with open(het_results_path, 'rb') as file:
            het_results = pickle.load(file)


        het_z = het_results['heter_trajs']
        het_log_lkh = mvn.log_prob(het_z).sum()
            
        het_trajs = get_trajs(model, obs_data, het_z, qmodel.mins, qmodel.maxs)

        baseline_results['het_z'] = het_z
        baseline_results['het_trajs'] = het_trajs
        baseline_results['het_log_lkh'] = het_log_lkh



    try:
        print('----> hom_log_lkh = ', hom_log_lkh)
    except:
        print("hom not done in this run")
    try:
        print('----> het_log_lkh = ', het_log_lkh)
    except:
        print("het not done in this run")
    try:
        print('----> guidance_log_lkh = ', guidance_log_lkh)
    except:
        print("guidance not done in this run")
    try:
        print('----> vanilla_log_lkh = ', vanilla_sat_log_lkh)
    except:
        print("vanilla_sat not done in this run")
    try:
        print("----> vanilla_log_lkh = ", vanilla_log_lkh)
    except:
        print("vanilla not done in this run, are you sure everything is correct?")

    baseline_path = os.path.join(parent_dir, 'Cert_algs',"results","DIFF",dataname,'EXP',"diff_"+dataname+"_"+ex_id+"_baseline_results.pkl")
    with open(baseline_path, "wb") as f:
        pickle.dump(baseline_results, f)
    
    numpy_data = {key: value.detach().cpu().numpy() for key, value in baseline_results.items()}

    # Salva il dizionario in un file MAT
    mat_path = os.path.join(parent_dir, 'Cert_algs',"results","DIFF",dataname,'EXP', "diff_"+dataname+"_"+ex_id+"_baseline_results.mat")
    scipy.io.savemat(mat_path, numpy_data)