import pickle

import os
import re
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset

import torch

class Dataset(Dataset):
	def __init__(self, model_name, target_dim, eval_length, missing_ratio=0.0, seed=0, datafolder='', idx='', scaling_flag=False, extra_info = '', train_ds = None, path= ''):
		self.eval_length = eval_length
		self.target_dim = target_dim
		np.random.seed(seed)  # seed for ground truth choice
		self.model_name = model_name
		self.observed_values = []
		self.observed_masks = []
		self.gt_masks = []
		self.extra_info = extra_info
		self.missing_ratio = missing_ratio

		

		if idx == 'test_fixed':
			observed_values, mask = self.build_mask(path+f'/data/'+datafolder+'/'+datafolder+'_data_test_fixed_froms.pickle', missing_ratio)
		else:	
			observed_values, mask = self.build_mask(path+f'/data/'+datafolder+'/'+datafolder+'_map_data_'+idx+'.pickle', missing_ratio)
			print('The shape of observed data for '+idx+ ' set is: ', observed_values.shape)
			print('The type of observed_value is ', type(observed_values))
			
		self.observed_values = observed_values
		self.observed_masks = np.ones(observed_values.shape)#mask
		self.gt_masks =  mask
		
		if scaling_flag:
			tmp_values = self.observed_values.reshape(-1, self.target_dim)
			tmp_masks = self.observed_masks.reshape(-1, self.target_dim)
			if idx == 'train':
				self.min = np.min(np.min(tmp_values, axis = 0),axis=0)
				self.max = np.max(np.max(tmp_values, axis = 0),axis=0)
			else:
				self.min = train_ds.min
				self.max = train_ds.max
			self.observed_values = (
				-1+2*(self.observed_values - self.min) / (self.max-self.min) * self.observed_masks
			)

	def __getitem__(self, org_index):
		index = org_index
		s = {
			"observed_data": self.observed_values[index],
			"observed_mask": self.observed_masks[index],
			"gt_mask": self.gt_masks[index],
			"timepoints": np.arange(self.eval_length),
		}
		return s

	def __len__(self):
		return len(self.observed_values)

	def build_mask(self, path, missing_ratio, index_list=None):
		
		with open(path, "rb") as f:
			data = pickle.load(f)

		long_trajs = data['trajs']
		print('long trajs shape is :', long_trajs.shape)
		full_trajs = long_trajs[:,::3,:]
		print('trajs shape now is :', full_trajs.shape)

		if index_list is not None:
			full_trajs = full_trajs[index_list]
		mask = np.zeros(full_trajs.shape)
		n_steps = full_trajs.shape[1]
		if missing_ratio < 0: # initial state is observed
			mask[:,:int(-missing_ratio)] = 1
			
		else:
			mask[:,:int(n_steps*(1-missing_ratio))] = 1

		#print("------------------", full_trajs.shape, mask.shape)
		return full_trajs, mask


	def build_dict_from_trajs(self, full_trajs):
		
		mask = torch.zeros_like(full_trajs)
		n_steps = full_trajs.shape[1]
		if self.missing_ratio < 0: # initial state is observed
			mask[:,:int(-self.missing_ratio)] = 1
		else:
			mask[:,:int(n_steps*(1-self.missing_ratio))] = 1

		obs_tp = torch.empty((full_trajs.shape[0],full_trajs.shape[1]))
		for b in range(full_trajs.shape[0]):
			obs_tp[b] = torch.arange(self.eval_length)
			
		#print("------------------", full_trajs.shape, mask.shape)

		s = {
			"observed_data": full_trajs,
			"observed_mask": torch.ones_like(full_trajs),
			"gt_mask": mask,
			"timepoints": obs_tp
		}
		return s


def get_dataloader(model_name, eval_length, target_dim, seed=1, nfold=None, batch_size=16, missing_ratio=0.1, scaling_flag = False, datafolder='', extra_info = '', path = ''):

	print('salerno', path)
	# only to obtain total length of dataset
	train_dataset = Dataset(model_name=model_name, target_dim=target_dim, eval_length=eval_length, missing_ratio=missing_ratio, seed=seed, datafolder=datafolder, idx='train', scaling_flag=scaling_flag, path = path)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)
	
	test_dataset = Dataset(model_name=model_name, target_dim=target_dim, eval_length=eval_length, missing_ratio=missing_ratio, seed=seed,datafolder=datafolder, idx='test', scaling_flag=scaling_flag, train_ds = train_dataset, path = path)
	test_loader = DataLoader(test_dataset, batch_size=16, shuffle=0)

	calibr_dataset = Dataset(model_name=model_name, target_dim=target_dim, eval_length=eval_length, missing_ratio=missing_ratio, seed=seed,datafolder=datafolder, idx='test', scaling_flag=scaling_flag, train_ds = train_dataset, path = path)
	calibr_loader = DataLoader(calibr_dataset, batch_size=16, shuffle=0)

	return train_loader, test_loader, calibr_loader


def get_train_dataloader(model_name, train_dataset, eval_length, target_dim, seed=1, nfold=None, batch_size=16, missing_ratio=0.1, scaling_flag = False, datafolder='', extra_info = ''):

	dataset = Dataset(model_name=model_name, target_dim=target_dim, eval_length=eval_length, missing_ratio=missing_ratio, seed=seed,datafolder=datafolder, idx='train', scaling_flag=scaling_flag)
	loader = DataLoader(dataset, batch_size=900, shuffle=0)

	return loader

def get_calibr_dataloader(model_name, train_dataset, eval_length, target_dim, seed=1, nfold=None, batch_size=16, missing_ratio=0.1, scaling_flag = False,datafolder='', extra_info = ''):

	calibr_dataset = Dataset(model_name=model_name, target_dim=target_dim, eval_length=eval_length, missing_ratio=missing_ratio, seed=seed,datafolder=datafolder, idx='test', scaling_flag=scaling_flag, train_ds = train_dataset, extra_info = extra_info)
	calibr_loader = DataLoader(calibr_dataset, batch_size=200, shuffle=0)

	return calibr_loader

def get_test_dataloader(model_name, train_dataset, eval_length, target_dim, seed=1, nfold=None, batch_size=16, missing_ratio=0.1, scaling_flag = False,datafolder='', extra_info = ''):

	calibr_dataset = Dataset(model_name=model_name, target_dim=target_dim, eval_length=eval_length, missing_ratio=missing_ratio, seed=seed,datafolder=datafolder, idx='test_fixed', scaling_flag=scaling_flag, train_ds = train_dataset, extra_info = extra_info)
	calibr_loader = DataLoader(calibr_dataset, batch_size=600, shuffle=0)

	return calibr_loader