import yaml
import scipy.io
import os
import math
import pickle
import sys
import gan_stl_models as cm
from utils_gan import *
sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

print(current_dir)
print(parent_dir)

from boolean_stl import *
import stl
from verification_cuda import *

import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *

from Models.GAN.generator import Generator
from Models.GAN.Dataset_Map import * 
cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

case_path = os.path.join(parent_dir,"Cert_algs","Cases","gan","crossroad.yaml")


with open(case_path, "r") as f:
    case_data = yaml.safe_load(f)


modelname = case_data['modelname']
dataname = case_data['dataname']

arch = 'GAN'
gen_id = case_data['gen_id']

traj_len = case_data['traj_len']
xdim = case_data['xdim']
latent_dim = case_data['latent_dim']
latent_shape = Tensor(np.empty((1, traj_len)))


genpath = os.path.join(parent_dir,'save', modelname, arch, gen_id, 'generator.pt' )

trainset_fn = os.path.join(parent_dir, 'data',dataname, dataname+'_map_data_train.pickle')
testset_fn = os.path.join(parent_dir, 'data',dataname,dataname+'_map_data_test.pickle')



ds = Dataset(trainset_fn, testset_fn, xdim, xdim, traj_len)
ds.load_train_data()
ds.load_test_data()

## NOTA : fix for test dimensions
Y = Tensor(ds.Y_train_transp[:1])

model = Generator(xdim, traj_len, latent_dim)
model = torch.load(genpath)
model.eval()

CondGenerator = getattr(cm, 'condGen_'+dataname)
CondGenerator_w_STL = getattr(cm, 'condGen_w_BoolSTL_'+dataname)
CondGenerator_w_QuantSTL = getattr(cm, 'condGen_w_QuantSTL_'+dataname)



cmodel = CondGenerator(model, Y)
cmodel.eval()

bmodel = CondGenerator_w_STL(model, Y)

qmodel = CondGenerator_w_QuantSTL(model, Y)

M_MAX = 20

EPS = 0.01
DELTA_EPS = 0.005
NSAMPLES = 200
WEIGHT = 0.1

CERTIFICATION = False
GEN_TRAJ = True
GUIDANCE = True
VANILLA_SAT = True

#for the generation purposes
ex_id = 'test'
HOM = True
HET = False

if HOM:
    cert_type='HOM'
elif HET:
    cert_type='HET'

ID = '_M20_H23'

hom_results_path = os.path.join(parent_dir, 'Cert_algs',"results","GAN",dataname,cert_type,'gan_'+dataname+'_Hom_results.pkl')
het_results_path = os.path.join(parent_dir, 'Cert_algs',"results","GAN",dataname,cert_type,'gan_'+dataname+'_Het_results.pkl')
hom_list_path = os.path.join(parent_dir, 'Cert_algs',"results","GAN",dataname,cert_type, "gan_"+dataname+"_Hom_list.pkl")
het_list_path = os.path.join(parent_dir, 'Cert_algs',"results","GAN",dataname,cert_type, "gan_"+dataname+"_Het_list.pkl")


if CERTIFICATION:
    if HOM:
        print('-------- HOMEGENEOUS INCREMENTS----------')
        #ball_list = verifier(bmodel, qmodel, M=10, epsilon = 0.1, latent_dim = latent_dim)
        hom_ball_list = verifier_increment(bmodel, qmodel, M=M_MAX, epsilon = EPS, delta_eps = DELTA_EPS, latent_shape = latent_shape)
        hom_balls = {"hom_ball_list": hom_ball_list}
        with open(hom_list_path, "wb") as f:
            pickle.dump(hom_balls, f)
        Mhom = len(hom_ball_list)
        print('nb of balls = ', Mhom)


        #hom_Z, hom_robs = empirical_satisfaction(hom_ball_list, qmodel, latent_dim, nsamples = NSAMPLES, model_name = dataname)

        #hom_log_prob,prob_v = compute_log_prob(hom_ball_list, latent_dim)

        #print('logprob = ', hom_log_prob)
        #print('vector of probs = ', prob_v)
        #print('sum of probs: ', np.sum(prob_v))
        
        #hom_results = {"hom_ball_list": hom_ball_list, "hom_log_prob": hom_log_prob}#, "hom_log_prob": hom_log_prob, 'hom_trajs': hom_Z, 'M_MAX': M_MAX, 'EPS': EPS, 'DELTA_EPS': DELTA_EPS, 'NSAMPLES': NSAMPLES,}
        
        #with open(hom_results_path, "wb") as f:
        #    pickle.dump(hom_results, f)

    if HET:
        print('-------- HETEROGENEOUS INCREMENTS----------')

        het_ball_list = verifier_heterog_increment(bmodel, qmodel, weight = WEIGHT, M=M_MAX, epsilon = EPS, delta_eps = DELTA_EPS, latent_shape = latent_shape)
        het_balls = {"het_ball_list": het_ball_list}
        with open(het_list_path, "wb") as f:
            pickle.dump(het_balls, f)
        Mhet = len(het_ball_list)
        print('nb of balls = ', Mhet)


        #heter_Z, heter_robs = empirical_satisfaction(heter_ball_list, qmodel, latent_dim, nsamples = NSAMPLES, model_name = 'maze heterogeneous')


        #heter_log_prob = compute_log_prob(heter_ball_list, latent_dim)

        #print('logprob = ', heter_log_prob)

        #het_results = {"heter_ball_list": het_ball_list}#, "heter_log_prob": heter_log_prob, 'heter_trajs': heter_Z, 'WEIGHT': WEIGHT, 'M_MAX': M_MAX, 'EPS': EPS, 'DELTA_EPS': DELTA_EPS, 'NSAMPLES': NSAMPLES }
        
        #with open(het_results_path, "wb") as f:
        #    pickle.dump(het_results, f)


if GEN_TRAJ:

    mean = torch.zeros(latent_dim, device=device)  
    covariance_matrix = torch.eye(latent_dim, device=device)  
    
    mvn = MultivariateNormal(mean, covariance_matrix)
 

    #baseline_path = os.path.join(parent_dir, 'Cert_algs',"results","GAN",dataname,"gan_"+dataname+"_baseline_results.pkl")
    vanilla_trajs = vanilla_samples(model, Y, NSAMPLES, latent_dim, qmodel.mins, qmodel.maxs)
    vanilla_sat_z = get_vanilla_sat_samples(qmodel, NSAMPLES, latent_dim)
    vanilla_log_lkh = mvn.log_prob(vanilla_sat_z)


    try:
        print("----> vanilla_log_lkh = ", vanilla_log_lkh)
    except:
        print("vanilla not done in this run, are you sure everything is correct?")

    baseline_results = {'vanilla_trajs': vanilla_trajs, 'vanilla_log_lkh' : vanilla_log_lkh,}

    if VANILLA_SAT:
        ex_id = ex_id + "_VS_"

        vanilla_sat_z = get_vanilla_sat_samples(qmodel, NSAMPLES, latent_dim)
        vanilla_sat_log_lkh = mvn.log_prob(vanilla_sat_z).sum()

        baseline_results['vanilla_sat_z'] = vanilla_sat_z
        baseline_results['vanilla_sat_log_lkh'] = vanilla_sat_log_lkh
        baseline_path = os.path.join(parent_dir, 'Cert_algs',"results","GAN",dataname,'EXP',"gan_"+dataname+"_"+ex_id+"_baseline_results.pkl")
        with open(baseline_path, "wb") as f:
            pickle.dump(baseline_results, f)



    if GUIDANCE:

        ex_id = ex_id +"_GUID_"
        guidance_trajs, guidance_z = guidance_samples(qmodel, latent_shape, NSAMPLES)
        guidance_log_lkh = mvn.log_prob(guidance_z).sum()

        baseline_results['guidance_z'] = guidance_z
        baseline_results['guidance_trajs'] = guidance_trajs
        baseline_results['guidance_log_lkh'] = guidance_log_lkh
        baseline_path = os.path.join(parent_dir, 'Cert_algs',"results","GAN",dataname,'EXP',"gan_"+dataname+"_"+ex_id+"_baseline_results.pkl")
        with open(baseline_path, "wb") as f:
            pickle.dump(baseline_results, f)


    if HOM:
        ex_id = ex_id + cert_type


        with open(hom_list_path, 'rb') as file:
            hom_ball_list_dict = pickle.load(file)

        hom_ball_list= hom_ball_list_dict["hom_ball_list"]
        hom_z, _ = empirical_satisfaction(hom_ball_list, qmodel, latent_dim, nsamples = NSAMPLES, model_name = dataname)
        hom_trajs = get_trajs(model, Y, hom_z, qmodel.mins, qmodel.maxs)
        hom_log_lkh = mvn.log_prob(hom_z).sum()
        hom_log_prob = compute_log_prob(hom_ball_list, latent_dim)

        print('logprob = ', hom_log_prob)

        hom_results = {"hom_ball_list": hom_ball_list, "hom_log_prob": hom_log_prob, 'hom_trajs': hom_z, 'M_MAX': M_MAX, 'EPS': EPS, 'DELTA_EPS': DELTA_EPS, 'NSAMPLES': NSAMPLES,}
        
        with open(hom_results_path, "wb") as f:
            pickle.dump(hom_results, f)

       

        with open(hom_results_path, 'rb') as file:
            hom_results = pickle.load(file)

        

        hom_z = hom_results['hom_trajs']
        hom_log_lkh = mvn.log_prob(hom_z).sum()

        baseline_results['hom_z'] = hom_z
        baseline_results['hom_trajs'] = hom_trajs
        baseline_results['hom_log_lkh'] = hom_log_lkh

        baseline_path = os.path.join(parent_dir, 'Cert_algs',"results","GAN",dataname,'EXP',"gan_"+dataname+"_"+ex_id+"_baseline_results.pkl")
        with open(baseline_path, "wb") as f:
            pickle.dump(baseline_results, f)
    
    if HET:

        ex_id = ex_id + cert_type


        with open(het_list_path, 'rb') as file:
            het_ball_list = pickle.load(file)
        het_z, _ = empirical_satisfaction(hom_results['het_ball_list'], qmodel, latent_dim, nsamples = NSAMPLES, model_name = dataname)
        het_trajs = get_trajs(model, Y, het_z, qmodel.mins, qmodel.maxs)
        het_log_lkh = mvn.log_prob(het_z).sum()
        het_log_prob = compute_log_prob(het_ball_list, latent_dim)

        print('logprob = ', het_log_prob)

        het_results = {"hom_ball_list": het_ball_list, "hom_log_prob": het_log_prob, 'hom_trajs': het_z, 'M_MAX': M_MAX, 'EPS': EPS, 'DELTA_EPS': DELTA_EPS, 'NSAMPLES': NSAMPLES,}
        
        with open(het_results_path, "wb") as f:
            pickle.dump(het_results, f)

       

        with open(het_results_path, 'rb') as file:
            het_results = pickle.load(file)

        
        het_log_lkh = mvn.log_prob(het_z).sum()

        baseline_results['het_z'] = het_z
        baseline_results['het_trajs'] = het_trajs
        baseline_results['het_log_lkh'] = het_log_lkh

        baseline_path = os.path.join(parent_dir, 'Cert_algs',"results","GAN",dataname,'EXP',"gan_"+dataname+"_"+ex_id+"_baseline_results.pkl")
        with open(baseline_path, "wb") as f:
            pickle.dump(baseline_results, f)



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

    baseline_path = os.path.join(parent_dir, 'Cert_algs',"results","GAN",dataname,'EXP',"gan_"+dataname+"_"+ex_id+"_baseline_results.pkl")
    with open(baseline_path, "wb") as f:
        pickle.dump(baseline_results, f)
    
    numpy_data = {key: value.detach().cpu().numpy() for key, value in baseline_results.items()}

    # Salva il dizionario in un file MAT
    mat_path = os.path.join(parent_dir, 'Cert_algs',"results","GAN",dataname,'EXP', "gan_"+dataname+"_"+ex_id+"_baseline_results.mat")
    scipy.io.savemat(mat_path, numpy_data)
    
    

