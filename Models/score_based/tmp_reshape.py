import pickle
import numpy as np
import sys
import os
sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import argparse
from model_details import *

parser = argparse.ArgumentParser(description="data_reshape")
parser.add_argument("--model_name", type=str, default='LV64')
parser.add_argument("--Q", type=int, default=50)
parser.add_argument("--active_flag", type=eval, default=False)
parser.add_argument("--model_id", type=int, default=1)
opt = parser.parse_args()

opt = get_model_details(opt)

indexes = np.arange(3,opt.traj_len+1,4)
    
# ---- TRAIN

train_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_train_set_H={opt.traj_len}_500x10.pickle'

with open(train_path, "rb") as f:
    train_dict = pickle.load(
        f
    )
print(train_dict['init'].shape,train_dict['trajs'].shape)


new_train_dict = {'trajs': train_dict['trajs'][:,indexes,:opt.x_dim], 'init': train_dict['init'][:,:,:opt.x_dim]}
print(new_train_dict['init'].shape,new_train_dict['trajs'].shape)

new_train_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_train_set_H={len(indexes)}_500x10.pickle'

with open(new_train_path, "wb") as f:
    pickle.dump(new_train_dict, f)

#------------ VALID

valid_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_valid_set_H={opt.traj_len}_100x50.pickle'

with open(valid_path, "rb") as f:
    valid_dict = pickle.load(
        f
    )

new_valid_dict = {'trajs': valid_dict['trajs'][:,indexes,:opt.x_dim], 'init': valid_dict['init'][:,:,:opt.x_dim]}

new_valid_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_valid_set_H={len(indexes)}_100x50.pickle'

with open(new_valid_path, "wb") as f:
    pickle.dump(new_valid_dict, f)


#----------------------- TEST

test_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_test_set_H={opt.traj_len}_25x1000.pickle'

with open(test_path, "rb") as f:
    test_dict = pickle.load(
        f
    )

new_test_dict = {'trajs': test_dict['trajs'][:,indexes,:opt.x_dim], 'init': test_dict['init'][:,:,:opt.x_dim]}

new_test_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_test_set_H={len(indexes)}_25x1000.pickle'

with open(new_test_path, "wb") as f:
    pickle.dump(new_test_dict, f)

# ---- ACTIVE

active_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_active_set_H={opt.traj_len}_200x10.pickle'

with open(active_path, "rb") as f:
    active_dict = pickle.load(
        f
    )

new_active_dict = {'trajs': active_dict['trajs'][:,indexes,:opt.x_dim], 'init': active_dict['init'][:,:,:opt.x_dim]}

new_active_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_active_set_H={len(indexes)}_200x10.pickle'

with open(new_active_path, "wb") as f:
    pickle.dump(new_active_dict, f)