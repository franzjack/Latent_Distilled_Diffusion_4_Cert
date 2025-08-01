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
        
        self.traj_len = traj_len


    def load_train_data(self):

        file = open(self.trainset_fn, 'rb')
        data = pickle.load(file)
        file.close()

        Y = np.expand_dims(np.array([traj[0] for traj in data["trajs"]]),1)

        X = np.array([traj[1:] for traj in data["trajs"]])

        
        print("TRAIN SHAPES: ", X.shape, Y.shape)

        self.HMAX = np.max(np.max(X, axis = 0),axis=0)
        print(self.HMAX)

        self.HMIN = np.min(np.min(X, axis = 0),axis=0)
        print(self.HMIN)
        # data scales between [-1,1]
        self.X_train = -1+2*(X-self.HMIN)/(self.HMAX-self.HMIN)
        self.Y_train = -1+2*(Y-self.HMIN)/(self.HMAX-self.HMIN)
        self.n_points_dataset = self.X_train.shape[0]

        self.X_train_transp = np.swapaxes(self.X_train,1,2)
        self.Y_train_transp = np.swapaxes(self.Y_train,1,2)    
        
        
         
        
        
    def load_test_data(self):

       
        file = open(self.testset_fn, 'rb')
        data = pickle.load(file)
        file.close()

        Y = np.expand_dims(np.array([traj[0] for traj in data["trajs"]]),1)

        X = np.array([traj[1:] for traj in data["trajs"]])

        print("TEST SHAPES: ", X.shape, Y.shape)

           
        self.n_points_test = X.shape[0]
        
        Xfl = -1+2*(X-self.HMIN)/(self.HMAX-self.HMIN)
        Yfl = -1+2*(Y-self.HMIN)/(self.HMAX-self.HMIN)
        

        self.X_test = Xfl
        self.Y_test = Yfl
        
        self.X_test_transp = np.swapaxes(self.X_test,1,2)
        self.Y_test_transp = np.swapaxes(self.Y_test,1,2)
        
        
 

    def generate_mini_batches(self, n_samples):
        file = open(self.trainset_fn, 'rb')
        data = pickle.load(file)
        file.close()

        ix = np.random.randint(0, self.X_train_transp.shape[0], n_samples)
        Xb = self.X_train_transp[ix]
        Yb = self.Y_train_transp[ix]
        
        return Xb, Yb
        
