import numpy as np
import math
import torch
from copy import copy
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import matplotlib.pyplot as plt
import pickle
import os
from scipy import stats

plt.rcParams.update({'font.size': 22})

torch.manual_seed(1)

cuda = True if torch.cuda.is_available() else False
cuda= False
device = 'cuda' if cuda else 'cpu'
print('verification device = ', device)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


## GAN--------------
def get_trajs(model, cond, latent_samples, mins, maxs):

	conds = Tensor(cond*torch.ones((latent_samples.shape[0], cond.shape[1], cond.shape[2]), device = device))

	scaled_trajs = model(conds,latent_samples)

	return mins + (scaled_trajs+1)*(maxs-mins)/2



def vanilla_samples(model, cond, nsamples, latent_dim, mins, maxs):

	conds = Tensor(cond*torch.ones((nsamples, cond.shape[1], cond.shape[2]), device = device))

	noises = Tensor(np.random.normal(0, 1, (nsamples, latent_dim)))

	scaled_trajs = model(conds,noises)

	return mins + (scaled_trajs+1)*(maxs-mins)/2



def get_vanilla_sat_samples(qmodel, nsamples, latent_dim):

	K = 1
	list_z = []

	tot_samples = 0 
	while len(list_z) < nsamples:
		
		ZZ = Tensor(np.random.normal(0, 1, (K, latent_dim)))

		RR = qmodel(ZZ)

		Zacc = ZZ[RR > 0]
		for i in range(len(Zacc)):
			list_z.append(Zacc[i])

		tot_samples += K

	print(f'Vanilla Acceptance Ratio: {nsamples}/{tot_samples} = {nsamples/tot_samples}')

	return torch.stack(list_z,dim=0).to(device)


def kolmogorov_smirnov_distance(samples):
	print('AAAA', samples.shape)
	latent_dim = samples.shape[1]

	statistics = np.empty(latent_dim)
	for d in range(latent_dim):
			ksres = stats.kstest(samples[:,d], 'norm')
			#ks_statistic, p_value
			statistics[d] = ksres.statistic
	#print(statistics)
	return np.mean(statistics)

## CSDI: TODO --------------



def diff_vanilla_samples(model, cond, nsamples, latent_shape, mins, maxs):

	conds = Tensor(cond*torch.ones((nsamples, cond.shape[1], cond.shape[2]), device = device))
	print('conds shape is: ', conds.shape)


	noises = torch.randn_like(conds[:,:,1:])
	print('noises shape is: ', noises.shape)


	scaled_trajs = model(conds,noises)

	return mins + (scaled_trajs+1)*(maxs-mins)/2, noises



def diff_get_vanilla_sat_samples(qmodel, nsamples, latent_shape):

	K = 1
	list_z = []

	tot_samples = 0 
	while len(list_z) < nsamples:
		
		ZZ = torch.randn_like(latent_shape)

		RR = qmodel(ZZ)

		Zacc = ZZ[RR > 0]
		for i in range(len(Zacc)):
			list_z.append(Zacc[i])

		tot_samples += K

	print(f'Vanilla Acceptance Ratio: {nsamples}/{tot_samples} = {nsamples/tot_samples}')

	return torch.stack(list_z,dim=0).to(device)

#def ball_collector(general_path, nfiles):
#	for i in range(nfiles):



## COMMON
def grad_ascent_w_trajs(qmodel, z0, lr= 0.005, tol = 10^(-4)):
	z0.requires_grad = True
	norm = np.inf
	z0 = torch.randn_like(z0)
	gradient = torch.ones_like(z0)
	z0.requires_grad = True
	c = 0
	while(torch.norm(gradient, norm)> tol and c < 500):
		z0.requires_grad = True
		robustness, traj = qmodel(z0, return_traj = True)
		
		gradient = torch.autograd.grad(robustness, z0, allow_unused=True)[0]
		with torch.no_grad():
			z0 = z0 + lr * gradient
		c += 1
	print('------------- optimal robustness = ', robustness)
	
	return traj, robustness, z0.detach() # x FRA: qmodel ritornava già traiettorie riscalate 


def guidance_samples(qmodel, latent_shape, nsamples):

	c = 0
	list_trajs = []
	list_zs = []
	tot_samples = 0
	while c < nsamples:

		z0 = torch.randn_like(latent_shape).to(device)
		traj, rob, z = grad_ascent_w_trajs(qmodel, z0)

		if rob > 0:
			list_trajs.append(traj)
			list_zs.append(z)
			c += 1
		tot_samples += 1

	print(f'Guidance Acceptance Ratio: {nsamples}/{tot_samples} = {nsamples/tot_samples}')
	return torch.stack(list_trajs, dim=0).to(device), torch.stack(list_zs, dim=0).to(device)
