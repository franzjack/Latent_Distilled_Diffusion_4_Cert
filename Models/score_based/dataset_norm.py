import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

import torch

import sys
import os

sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class Dataset(Dataset):
	def __init__(self, model_name, target_dim, eval_length, missing_ratio=0.0, seed=0, idx='', scaling_flag=False, retrain_id = '', train_ds = None):
		self.eval_length = eval_length
		self.target_dim = target_dim
		np.random.seed(seed)  # seed for ground truth choice
		self.model_name = model_name
		self.observed_values = []
		self.observed_masks = []
		self.gt_masks = []
		self.missing_ratio = missing_ratio
		use_index_list = None
		if scaling_flag:
			newpath = (f"../data/{model_name}/{model_name}_norm_"+idx+f"_missing{missing_ratio}_gtmask.pickle")
		else:
			newpath = (f"../data/{model_name}/{model_name}_"+idx+f"_missing{missing_ratio}_gtmask.pickle")
		if os.path.isfile(newpath) == False:  # if datasetfile is none, create

			if model_name =="MAPK":
				if idx == 'test':
					observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_test_trajs_H={eval_length}_40x1000.pickle', missing_ratio, use_index_list)
				elif idx == 'valid':
					observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_valid_trajs_H={eval_length}_500x50.pickle', missing_ratio, use_index_list)
				else:
					if retrain_id == '':
						observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_train_trajs_H={eval_length}_500x50.pickle', missing_ratio, use_index_list)
					else:
						observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_csdi{retrain_id}_{50}perc_retrain_set_H=_3000x50.pickle', missing_ratio, use_index_list)

			else:
				if idx == 'test':
					observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_test_trajs_H={eval_length}_40x1000.pickle', missing_ratio, use_index_list)

				elif idx == 'valid':
					if model_name =="LV" or model_name =="LV64":
						observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_valid_trajs_H={eval_length}_500x50.pickle', missing_ratio, use_index_list)
					else:
						observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_valid_trajs_H={eval_length}_500x50.pickle', missing_ratio, use_index_list)


				else:
					if retrain_id == '':
						if model_name =="LV" or model_name =="LV64":
							observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_train_trajs_H={eval_length}_500x50.pickle', missing_ratio, use_index_list)
						else:
							observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_train_trajs_H={eval_length}_500x50.pickle', missing_ratio, use_index_list)
					else:
						observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_csdi{retrain_id}_{50}perc_retrain_set_H=_1000x10.pickle', missing_ratio, use_index_list)

			self.observed_values = observed_values
			self.observed_masks = np.ones(observed_values.shape)#mask
			self.gt_masks =  mask
			# calc mean and std and normalize values
			# (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
			
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

			with open(newpath, "wb") as f:
				pickle.dump(
					[self.observed_values, self.observed_masks, self.gt_masks, self.min, self.max], f
				)
		else:  # load datasetfile
			with open(newpath, "rb") as f:
				self.observed_values, self.observed_masks, self.gt_masks, self.min, self.max = pickle.load(
					f
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
			datadict = pickle.load(f)
		full_trajs = datadict['trajs']
		if index_list is not None:
			full_trajs = full_trajs[index_list]
		mask = np.zeros(full_trajs.shape)
		n_steps = full_trajs.shape[1]
		if missing_ratio < 0: #only initial state is observed
			mask[:,:int(-missing_ratio)] = 1
		else:
			mask[:,:int(n_steps*(1-missing_ratio))] = 1

		print("------------------", full_trajs.shape, mask.shape)
		return full_trajs, mask


	def build_dict_from_trajs(self, full_trajs):
		
		mask = torch.zeros_like(full_trajs)
		n_steps = full_trajs.shape[1]
		if self.missing_ratio < 0: #only initial state is observed
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


def get_dataloader(model_name, eval_length, target_dim, seed=1, nfold=None, batch_size=16, missing_ratio=0.1, scaling_flag = False, retrain_id = ''):

	# only to obtain total length of dataset
	train_dataset = Dataset(model_name=model_name, target_dim=target_dim, eval_length=eval_length, missing_ratio=missing_ratio, seed=seed, idx='train', scaling_flag=scaling_flag, retrain_id = retrain_id)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)
	
	valid_dataset = Dataset(model_name=model_name, target_dim=target_dim, eval_length=eval_length, missing_ratio=missing_ratio, seed=seed, idx='valid', scaling_flag=scaling_flag, train_ds = train_dataset)
	valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=0)

	test_dataset = Dataset(model_name=model_name, target_dim=target_dim, eval_length=eval_length, missing_ratio=missing_ratio, seed=seed, idx='test', scaling_flag=scaling_flag, train_ds = train_dataset)
	test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=0)

	
	return train_loader, valid_loader, test_loader
