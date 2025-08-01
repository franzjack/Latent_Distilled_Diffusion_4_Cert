from boolean_stl import *
import stl
import os
import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
#from Models.score_based.new_main import Generator, diff_CSDI, absCSDI

cuda=False
device = 'cuda' if cuda else 'cpu'

print('exec device = ', device)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class condGen(nn.Module):
    def __init__(self, generator, obs_data,MAXES, MINS):
        super(condGen, self).__init__()

        self.generator = generator.to(device)
        self.data = obs_data.to(device)
        self.maxs = torch.tensor(MAXES).to(device)
        self.mins = torch.tensor(MINS).to(device) 

    def forward(self, noise):

        signal = self.generator(self.data, noise).to(device)
        rescaled_signal = self.mins + (signal+1)*(self.maxs-self.mins)/2
        
        #rescaled_signal = ((signal.permute(0,2,1)+1)*self.maxs/2).permute(0,2,1)

        return rescaled_signal

#-----------MAZE MODELS--------------------



class condGen_w_QuantSTL_maze(nn.Module):
    def __init__(self, generator, obs_data,MAXES, MINS):
        super(condGen_w_QuantSTL_maze, self).__init__()

        self.generator = generator
        self.obs_data = obs_data.to(device) 
        self.C1, self.R1 = [[25.,25.]], 5.
        self.C2, self.R2 = [[25., 15.]], 5.

        safety_1 = stl.Atom(var_index=2, threshold=self.R1, lte = False)
        safety_2 = stl.Atom(var_index=3, threshold=self.R2, lte = False)
        
        safe_and = stl.And(safety_1, safety_2)
        

        

        atom_xu = stl.Atom(var_index=0, threshold=45, lte=True)
        atom_xl = stl.Atom(var_index=0, threshold=5, lte=False)
        atom_yu = stl.Atom(var_index=1, threshold=45, lte=True)
        atom_yl = stl.Atom(var_index=1, threshold=5, lte=False)

        and_x = stl.And(atom_xl, atom_xu)
        and_y = stl.And(atom_yl, atom_yu)
        and_xy = stl.And(and_x, and_y)

        and_all = stl.And(and_xy, safe_and)
        

        self.formula = glob = stl.Globally(and_all, unbound=True)
        self.maxs = torch.tensor(MAXES).to(device)
        self.mins = torch.tensor(MINS).to(device)  

    def signal_norm(self, x, center):

        return torch.norm(x-center.unsqueeze(2),np.inf, dim=1).unsqueeze(1).to(device)

    def forward(self, noise, return_traj = False):


        signal = self.generator(noise, self.obs_data).to(device)
        rescaled_signal = self.mins + (signal+1)*(self.maxs-self.mins)/2
        #self.mins.unsqueeze(1).unsqueeze(0)+((signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)

        sign_obs1 = self.signal_norm(rescaled_signal,torch.tensor(self.C1,device=device))
        sign_obs2 = self.signal_norm(rescaled_signal,torch.tensor(self.C2,device=device))
        

        concat_signal = torch.cat((rescaled_signal, sign_obs1, sign_obs2), dim=1)

        if return_traj:
            return self.formula.quantitative(concat_signal), rescaled_signal
        else:
            return self.formula.quantitative(concat_signal)





class condGen_w_BoolSTL_maze(nn.Module):
    def __init__(self, generator, obs_data,MAXES, MINS):
        super(condGen_w_BoolSTL_maze, self).__init__()

        self.generator = generator
        self.obs_data = obs_data.to(device)
        self.C1, self.R1 = [[25.,25.]], 5.
        self.C2, self.R2 = [[25., 15.]], 5.
        
        def my_l2_norm(x):

            return torch.sqrt((x**2).sum(dim=1)).to(device)



        def atomic_predicate(x, ind, thresh, lte=True):
            if lte:
                return torch.sign(torch.relu(thresh-x[:,ind])).to(device)
            else:
                return torch.sign(torch.relu(x[:,ind]-thresh)).to(device)

        def atomic_predicate_norm(x, center, radius, lte=True):

            if lte:
                return torch.sign(torch.relu(radius-my_l2_norm(x-center)))
                
            else:
                return torch.sign(torch.relu(my_l2_norm(x-center)-radius))
                


        safety_1 = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(self.C1,device=device).unsqueeze(2), radius = torch.tensor(self.R1,device=device), lte=False)
        safety_2 = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(self.C2,device=device).unsqueeze(2), radius = torch.tensor(self.R2,device=device), lte=False)
        
        
        safe_and = bAnd(safety_1.evaluate, safety_2.evaluate)
        
        

        # x < 15 and 27 < y < 35
        atom_xl = bAtomicPredicate(atomic_predicate, ind=0, thresh = 5, lte=False)
        atom_xu = bAtomicPredicate(atomic_predicate, ind=0, thresh = 45, lte=True)
        
        atom_yl = bAtomicPredicate(atomic_predicate, ind=1, thresh = 5, lte=False)
        atom_yu = bAtomicPredicate(atomic_predicate, ind=1, thresh = 45, lte=True)
        
        and_x = bAnd(atom_xl.evaluate, atom_xu.evaluate)
        and_y = bAnd(atom_yl.evaluate, atom_yu.evaluate)
        and_xy = bAnd(and_x.evaluate, and_y.evaluate)
        
        and_all = bAnd(and_xy.evaluate, safe_and.evaluate)

        self.formula = bAlways(and_all.evaluate)
        self.maxs = torch.tensor(MAXES).to(device)
        self.mins = torch.tensor(MINS).to(device)

    def forward(self, noise):

        signal = self.generator(self.obs_data,noise)
        signal = signal.to(device)
        rescaled_signal = self.mins + (signal+1)*(self.maxs-self.mins)/2

        #rescaled_signal = self.mins.unsqueeze(1).unsqueeze(0)+((signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)

        return self.formula.evaluate(rescaled_signal).to(device)[:,0]
    


#---------OBS MODELS---------------





class condGen_w_QuantSTL_obs(nn.Module):
    def __init__(self, generator, obs_data,MAXES,MINS):
        super(condGen_w_QuantSTL_obs, self).__init__()

        self.generator = generator
        self.obs_data = obs_data.to(device) 

        self.C1, self.R1 = [[7.5,22.5]], 2.5
        self.C2, self.R2 = [[13., 13.]], 3.
        self.C3, self.R3 = [[22.5,7.5]], 2.5
        self.C4, self.R4 = [[19.,21.]], 2.

        safety_1 = stl.Atom(var_index=2, threshold=self.R1, lte = False)
        safety_2 = stl.Atom(var_index=3, threshold=self.R2, lte = False)
        safety_3 = stl.Atom(var_index=4, threshold=self.R3, lte = False)
        safety_4 = stl.Atom(var_index=5, threshold=self.R4, lte = False)
        
        and_12 = stl.And(safety_1, safety_2)
        and_34 = stl.And(safety_3, safety_4)
        safe_and = stl.And(and_12, and_34)

        

        atom_xu = stl.Atom(var_index=0, threshold=30, lte=True)
        atom_xl = stl.Atom(var_index=0, threshold=0, lte=False)
        atom_yu = stl.Atom(var_index=1, threshold=30, lte=True)
        atom_yl = stl.Atom(var_index=1, threshold=0, lte=False)

        and_x = stl.And(atom_xl, atom_xu)
        and_y = stl.And(atom_yl, atom_yu)
        and_xy = stl.And(and_x, and_y)

        and_all = stl.And(and_xy, safe_and)
        

        self.formula = glob = stl.Globally(and_all, unbound=True)
        self.maxs = torch.tensor(MAXES).to(device)
        self.mins = torch.tensor(MINS).to(device)  

    def signal_norm(self, x, center):

        return torch.norm(x-center.unsqueeze(2),np.inf, dim=1).unsqueeze(1).to(device)

    def forward(self, noise, return_traj = False):


        signal = self.generator(noise, self.obs_data).to(device)
        rescaled_signal = self.mins + (signal+1)*(self.maxs-self.mins)/2
        #self.mins.unsqueeze(1).unsqueeze(0)+((signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)

        sign_obs1 = self.signal_norm(rescaled_signal,torch.tensor(self.C1,device=device))
        sign_obs2 = self.signal_norm(rescaled_signal,torch.tensor(self.C2,device=device))
        sign_obs3 = self.signal_norm(rescaled_signal,torch.tensor(self.C3,device=device))
        sign_obs4 = self.signal_norm(rescaled_signal,torch.tensor(self.C4,device=device))

        concat_signal = torch.cat((rescaled_signal, sign_obs1, sign_obs2, sign_obs3, sign_obs4), dim=1)

        if return_traj:
            return self.formula.quantitative(concat_signal), rescaled_signal
        else:
            return self.formula.quantitative(concat_signal)





class condGen_w_BoolSTL_obs(nn.Module):
    def __init__(self, generator, obs_data,MAXES,MINS):
        super(condGen_w_BoolSTL_obs, self).__init__()

        self.generator = generator
        self.obs_data = obs_data.to(device)
        self.C1, self.R1 = [[7.5,22.5]], 2.5
        self.C2, self.R2 = [[13., 13.]], 3.
        self.C3, self.R3 = [[22.5,7.5]], 2.5
        self.C4, self.R4 = [[19.,21.]], 2.
        
        def my_l2_norm(x):

            return torch.sqrt((x**2).sum(dim=1)).to(device)



        def atomic_predicate(x, ind, thresh, lte=True):
            if lte:
                return torch.sign(torch.relu(thresh-x[:,ind])).to(device)
            else:
                return torch.sign(torch.relu(x[:,ind]-thresh)).to(device)

        def atomic_predicate_norm(x, center, radius, lte=True):

            if lte:
                return torch.sign(torch.relu(radius-my_l2_norm(x-center)))
                
            else:
                return torch.sign(torch.relu(my_l2_norm(x-center)-radius))
                


        safety_1 = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(self.C1,device=device).unsqueeze(2), radius = torch.tensor(self.R1,device=device), lte=False)
        safety_2 = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(self.C2,device=device).unsqueeze(2), radius = torch.tensor(self.R2,device=device), lte=False)
        safety_3 = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(self.C3,device=device).unsqueeze(2), radius = torch.tensor(self.R3,device=device), lte=False)
        safety_4= bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(self.C4,device=device).unsqueeze(2), radius = torch.tensor(self.R4,device=device), lte=False)
        
        and_12 = bAnd(safety_1.evaluate, safety_2.evaluate)
        and_34 = bAnd(safety_3.evaluate, safety_4.evaluate)

        safe_and = bAnd(and_12.evaluate, and_34.evaluate)
        

        # x < 15 and 27 < y < 35
        atom_xl = bAtomicPredicate(atomic_predicate, ind=0, thresh = 0, lte=False)
        atom_xu = bAtomicPredicate(atomic_predicate, ind=0, thresh = 30, lte=True)
        
        atom_yl = bAtomicPredicate(atomic_predicate, ind=1, thresh = 0, lte=False)
        atom_yu = bAtomicPredicate(atomic_predicate, ind=1, thresh = 30, lte=True)
        
        and_x = bAnd(atom_xl.evaluate, atom_xu.evaluate)
        and_y = bAnd(atom_yl.evaluate, atom_yu.evaluate)
        and_xy = bAnd(and_x.evaluate, and_y.evaluate)
        
        and_all = bAnd(and_xy.evaluate, safe_and.evaluate)

        self.formula = bAlways(and_all.evaluate)
        self.maxs = torch.tensor(MAXES).to(device)
        self.mins = torch.tensor(MINS).to(device)

    def forward(self, noise):

        signal = self.generator(self.obs_data,noise)
        signal = signal.to(device)
        rescaled_signal = self.mins + (signal+1)*(self.maxs-self.mins)/2

        #rescaled_signal = self.mins.unsqueeze(1).unsqueeze(0)+((signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)

        return self.formula.evaluate(rescaled_signal).to(device)[:,0] 
    


#----------------HCROSS MODELS-----------------

class condGen_w_QuantSTL_half_cross(nn.Module):
    def __init__(self, generator, obs_data,MAXES,MINS):
        super(condGen_w_QuantSTL_half_cross, self).__init__()

        self.generator = generator.to(device)
        self.data = obs_data.to(device)
        
        # for CROSSROAD Eventually(Globally (X< thresh))
        # x < 15 and 27 < y < 35
        atom_x = stl.Atom(var_index=0, threshold=15, lte=True)
        atom_yu = stl.Atom(var_index=1, threshold=35, lte=True)
        atom_yl = stl.Atom(var_index=1, threshold=27, lte=False)

        and_y = stl.And(atom_yl, atom_yu)
        and_xy = stl.And(and_y, atom_x)
        glob = stl.Globally(and_xy, unbound=True)

        self.formula = stl.Eventually(glob, unbound=True, time_bound=30)

        self.maxs = torch.tensor(MAXES).to(device) 
        self.mins = torch.tensor(MINS).to(device)

    def forward(self, noise, return_traj = False):

        signal = self.generator(self.data, noise).to(device)
        
       
        #rescaled_signal = ((signal.permute(0,2,1)+1)*self.maxs/2).permute(0,2,1)
        rescaled_signal = (signal+1)*self.maxs/2
        if return_traj:
            return self.formula.quantitative(rescaled_signal), rescaled_signal
        else:
            return self.formula.quantitative(rescaled_signal)




class condGen_w_BoolSTL_half_cross(nn.Module):
    def __init__(self, generator, obs_data,MAXES,MINS):
        super(condGen_w_BoolSTL_half_cross, self).__init__()

        self.generator = generator
        self.obsdata = obs_data
        
        def atomic_predicate(x, ind, thresh, lte=True):
            if lte:
                return torch.sign(torch.relu(thresh-x[:,ind])).to(device)
            else:
                return torch.sign(torch.relu(x[:,ind]-thresh)).to(device)

        # x < 15 and 27 < y < 35
        atom_x = bAtomicPredicate(atomic_predicate, ind=0, thresh = 15)
        atom_yu = bAtomicPredicate(atomic_predicate, ind=1, thresh = 35)
        atom_yl = bAtomicPredicate(atomic_predicate, ind=1, thresh = 27, lte=False)

        and_y = bAnd(atom_yl.evaluate, atom_yu.evaluate)
        and_xy = bAnd(and_y.evaluate, atom_x.evaluate)
        glob = bAlways(and_xy.evaluate)

        self.formula = bEventually(glob.evaluate_time)

        self.maxs = torch.tensor(MAXES).to(device) 
        self.mins = torch.tensor(MINS).to(device)

    def forward(self, noise):

        signal = self.generator(self.obsdata,noise)
        signal = signal.to(device)

        #rescaled_signal = ((signal.permute(0,2,1)+1)*self.maxs/2).permute(0,2,1)
        rescaled_signal = (signal+1)*self.maxs/2

        return self.formula.evaluate(rescaled_signal).to(device)