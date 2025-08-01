import numpy as np
import math
import sys

import pickle

class Dataset(object):

    def __init__(self, trainset_fn, testset_fn, x_dim, y_dim, traj_len):
        self.trainset_fn = trainset_fn
        self.testset_fn = testset_fn
        self.x_dim = x_dim 
        self.y_dim = y_dim 
        self.z_dim = y_dim
        self.traj_len = traj_len

    def add_valid_data(self, validset_fn):
        self.validset_fn = validset_fn

    def add_active_data(self, activeset_fn):
        self.activeset_fn = activeset_fn

    def load_train_data(self):

        file = open(self.trainset_fn, 'rb')
        data = pickle.load(file)
        file.close()

        X = data["trajs"][:,:self.traj_len,:]
        Y = data["init"]
        print("DATASET SHAPES: ", X.shape, Y.shape)

        self.HMAX = np.max(np.max(X, axis = 0),axis=0)

        self.HMIN = np.min(np.min(X, axis = 0),axis=0)

        # data scales between [-1,1]
        self.X_train = -1+2*(X-self.HMIN)/(self.HMAX-self.HMIN)
        self.Y_train = -1+2*(Y-self.HMIN)/(self.HMAX-self.HMIN)
        self.n_points_dataset = self.X_train.shape[0]

        self.X_train_transp = np.swapaxes(self.X_train,1,2)
        self.Y_train_transp = np.swapaxes(self.Y_train,1,2)    
        if "goal" in list(data):
            Z = data["goal"]
            self.Z_train =  -1+2*(Z-self.HMIN)/(self.HMAX-self.HMIN)
            self.Z_train_transp = np.swapaxes(self.Z_train,1,2)  
        
        
    def load_test_data(self, nb_trajs_per_point=1):

        self.n_test_traj_per_point = nb_trajs_per_point

        file = open(self.testset_fn, 'rb')
        data = pickle.load(file)
        file.close()

        X = data["trajs"][:,:self.traj_len,:]
        Y = data["init"]        
        self.n_points_test = X.shape[0]//nb_trajs_per_point
        
        Xfl = -1+2*(X-self.HMIN)/(self.HMAX-self.HMIN)
        Yfl = -1+2*(Y-self.HMIN)/(self.HMAX-self.HMIN)
        

        self.X_test = Xfl.reshape((self.n_points_test, self.n_test_traj_per_point, self.traj_len, self.x_dim))
        self.Y_test = Yfl.reshape((self.n_points_test, self.n_test_traj_per_point, 1, self.y_dim))
        
        self.X_test_transp = np.swapaxes(self.X_test,2,3)
        self.Y_test_transp = np.swapaxes(self.Y_test,2,3)
        if "goal" in list(data):
            Z = data["goal"]
            Zfl =  -1+2*(Z-self.HMIN)/(self.HMAX-self.HMIN)
            self.Z_test = Zfl.reshape((self.n_points_test, self.n_test_traj_per_point, 1, self.z_dim))
            self.Z_test_transp = np.swapaxes(self.Z_test,2,3) 
        

    def load_valid_data(self, nb_trajs_per_point=1):

        self.n_valid_traj_per_point = nb_trajs_per_point

        file = open(self.validset_fn, 'rb')
        data = pickle.load(file)
        file.close()

        X = data["trajs"][:,:self.traj_len,:]
        Y = data["init"]        
        self.n_points_valid = X.shape[0]//nb_trajs_per_point
        
        Xfl = -1+2*(X-self.HMIN)/(self.HMAX-self.HMIN)
        Yfl = -1+2*(Y-self.HMIN)/(self.HMAX-self.HMIN)
        

        self.X_valid = Xfl.reshape((self.n_points_valid, self.n_valid_traj_per_point, self.traj_len, self.x_dim))
        self.Y_valid = Yfl.reshape((self.n_points_valid, self.n_valid_traj_per_point, 1, self.y_dim))
        
        self.X_valid_transp = np.swapaxes(self.X_valid,2,3)
        self.Y_valid_transp = np.swapaxes(self.Y_valid,2,3)
        if "goal" in list(data):
            Z = data["goal"]
            Zfl =  -1+2*(Z-self.HMIN)/(self.HMAX-self.HMIN)
            self.Z_valid = Zfl.reshape((self.n_points_valid, self.n_valid_traj_per_point, 1, self.z_dim))
            self.Z_valid_transp = np.swapaxes(self.Z_valid,2,3) 

    def load_active_data(self, nb_trajs_per_point=10):

        self.n_active_traj_per_point = nb_trajs_per_point

        file = open(self.activeset_fn, 'rb')
        data = pickle.load(file)
        file.close()

        X = data["trajs"][:,:self.traj_len,:]
        Y = data["init"]        
        self.n_points_active = X.shape[0]//nb_trajs_per_point
        
        Xfl = -1+2*(X-self.HMIN)/(self.HMAX-self.HMIN)
        Yfl = -1+2*(Y-self.HMIN)/(self.HMAX-self.HMIN)
        

        self.X_active = Xfl.reshape((self.n_points_active, self.n_active_traj_per_point, self.traj_len, self.x_dim))
        self.Y_active = Yfl.reshape((self.n_points_active, self.n_active_traj_per_point, 1, self.y_dim))
        
        self.X_active_transp = np.swapaxes(self.X_active,2,3)
        self.Y_active_transp = np.swapaxes(self.Y_active,2,3)


    def generate_mini_batches(self, n_samples):
        file = open(self.validset_fn, 'rb')
        data = pickle.load(file)
        file.close()

        ix = np.random.randint(0, self.X_train_transp.shape[0], n_samples)
        Xb = self.X_train_transp[ix]
        Yb = self.Y_train_transp[ix]
        if "goal" in list(data):
            Zb = self.Z_train_transp[ix]
            return Xb, Yb, Zb
        else:
            return Xb, Yb
