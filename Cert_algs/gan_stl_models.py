from boolean_stl import *
import stl
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import math
sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



#-------------MAZE--------------------------


MAXES_m = [43.99893003, 44.98258877]
MINS_m = [5.00268656, 6.0001636 ]

OBS_m1 = [[25., 25.]]
OBS_m2 = [[25., 15.]]

R1_m, R2_m = 5.*math.sqrt(2.), 5.*math.sqrt(2.)

class condGen_maze(nn.Module):
    def __init__(self, generator, cond):
        super(condGen_maze, self).__init__()

        self.generator = generator
        self.cond = cond.to(device)
        self.maxs = torch.tensor(MAXES_m).to(device)
        self.mins = torch.tensor(MINS_m).to(device)

    def forward(self, noise):

        signal = self.generator(noise, self.cond).to(device)
        rescaled_signal = self.mins.unsqueeze(1).unsqueeze(0)+((signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)

        return rescaled_signal


class condGen_w_BoolSTL_maze(nn.Module):
    def __init__(self, generator, cond):
        super(condGen_w_BoolSTL_maze, self).__init__()

        self.generator = generator
        self.cond = cond.to(device)

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
                
        atom_1 = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(OBS_m1,device=device).unsqueeze(2), radius = torch.tensor(R1_m,device=device), lte=False)
        atom_2 = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(OBS_m2,device=device).unsqueeze(2), radius = torch.tensor(R2_m,device=device), lte=False)
        
        atom_12 = bAnd(atom_1.evaluate, atom_2.evaluate)
        
        atom_xl = bAtomicPredicate(atomic_predicate, ind = 0, thresh = 5, lte=False)
        atom_xu = bAtomicPredicate(atomic_predicate, ind = 0, thresh = 45, lte=True)
        atom_yl = bAtomicPredicate(atomic_predicate, ind = 1, thresh = 5, lte=False)
        atom_yu = bAtomicPredicate(atomic_predicate, ind = 1, thresh = 45, lte=True)
        
        atom_x = bAnd(atom_xl.evaluate, atom_xu.evaluate)
        atom_y = bAnd(atom_yl.evaluate, atom_yu.evaluate)
        atom_xy = bAnd(atom_x.evaluate, atom_y.evaluate)
        
        atom_all = bAnd(atom_xy.evaluate, atom_12.evaluate)
        self.formula = bAlways(atom_all.evaluate)
        #self.formula = bAlways(atom_12.evaluate)

        self.maxs = torch.tensor(MAXES_m).to(device) 
        self.mins = torch.tensor(MINS_m).to(device)

    def forward(self, noise):

        signal = self.generator(noise, self.cond).to(device)

        rescaled_signal = self.mins.unsqueeze(1).unsqueeze(0)+((signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)

        return self.formula.evaluate(rescaled_signal).to(device) 


class condGen_w_QuantSTL_maze(nn.Module):
    def __init__(self, generator, cond):
        super(condGen_w_QuantSTL_maze, self).__init__()

        self.generator = generator
        self.cond = cond.to(device) 
        
        # for CROSSROAD Eventually(Globally (X< thresh))

        atom_1 = stl.Atom(var_index=2, threshold=R1_m, lte=False)
        atom_2 = stl.Atom(var_index=3, threshold=R2_m, lte=False)

        atom_12 = stl.And(atom_1, atom_2)
        
        atom_xl = stl.Atom(var_index=0, threshold=5., lte=False)
        atom_xu = stl.Atom(var_index=0, threshold=45., lte=True)
        atom_yl = stl.Atom(var_index=1, threshold=5., lte=False)
        atom_yu = stl.Atom(var_index=1, threshold=45., lte=True)


        atom_x = stl.And(atom_xl, atom_xu)
        atom_y = stl.And(atom_yl, atom_yu)
        atom_xy = stl.And(atom_x, atom_y)

        atom_all = stl.And(atom_xy, atom_12)
 
        self.formula = stl.Globally(atom_all, unbound=True)

        self.maxs = torch.tensor(MAXES_m).to(device)  ### TODO: set the right values here
        self.mins = torch.tensor(MINS_m).to(device)

    def signal_norm(self, x, center):

        return torch.norm(x-center.unsqueeze(2),np.inf, dim=1).unsqueeze(1).to(device)

    def forward(self, noise, return_traj = False):

        signal = self.generator(noise, self.cond).to(device)
        rescaled_signal = self.mins.unsqueeze(1).unsqueeze(0)+((signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)

        obs1_sign = self.signal_norm(rescaled_signal,torch.tensor(OBS_m1,device=device))
        obs2_sign = self.signal_norm(rescaled_signal,torch.tensor(OBS_m2,device=device))
        
        concat_signal = torch.cat((rescaled_signal, obs1_sign, obs2_sign), dim=1)

        if return_traj:
            return self.formula.quantitative(concat_signal), rescaled_signal
        else:
            return self.formula.quantitative(concat_signal)




#---------OBS-----------------    





MAXES_obs = [29.99990511, 29.99987473]
MINS_obs = [1.82846450e-04, 6.78010084e-05]

C1_o, R1_o = [[7.5,22.5]], 2.5*math.sqrt(2.)
C2_o, R2_o = [[13., 13.]], 3.*math.sqrt(2.)
C3_o, R3_o = [[22.5,7.5]], 2.5*math.sqrt(2.)
C4_o, R4_o = [[19.,21.]], 2.*math.sqrt(2.)

class condGen_obs(nn.Module):
    def __init__(self, generator, cond):
        super(condGen_obs, self).__init__()

        self.generator = generator
        self.cond = cond.to(device)
        self.maxs = torch.tensor(MAXES_obs).to(device) 
        self.mins = torch.tensor(MINS_obs).to(device)
    
    def forward(self, noise):

        signal = self.generator(noise, self.cond).to(device)
        rescaled_signal = self.mins.unsqueeze(1).unsqueeze(0)+((signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)

        return rescaled_signal
    




class condGen_w_BoolSTL_obs(nn.Module):
    def __init__(self, generator, cond):
        super(condGen_w_BoolSTL_obs, self).__init__()

        self.generator = generator
        self.cond = cond.to(device)
        
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
                


        safety_1 = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(C1_o,device=device).unsqueeze(2), radius = torch.tensor(R1_o,device=device), lte=False)
        safety_2 = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(C2_o,device=device).unsqueeze(2), radius = torch.tensor(R2_o,device=device), lte=False)
        safety_3 = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(C3_o,device=device).unsqueeze(2), radius = torch.tensor(R3_o,device=device), lte=False)
        safety_4 = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(C4_o,device=device).unsqueeze(2), radius = torch.tensor(R4_o,device=device), lte=False)
        
        and_12 = bAnd(safety_1.evaluate, safety_2.evaluate)
        and_34 = bAnd(safety_3.evaluate, safety_4.evaluate)

        safe_and = bAnd(and_12.evaluate, and_34.evaluate)
        

        atom_xl = bAtomicPredicate(atomic_predicate, ind=0, thresh = 0, lte=False)
        atom_xu = bAtomicPredicate(atomic_predicate, ind=0, thresh = 30, lte=True)
        
        atom_yl = bAtomicPredicate(atomic_predicate, ind=1, thresh = 0, lte=False)
        atom_yu = bAtomicPredicate(atomic_predicate, ind=1, thresh = 30, lte=True)
        
        and_x = bAnd(atom_xl.evaluate, atom_xu.evaluate)
        and_y = bAnd(atom_yl.evaluate, atom_yu.evaluate)
        and_xy = bAnd(and_x.evaluate, and_y.evaluate)
        
        and_all = bAnd(and_xy.evaluate, safe_and.evaluate)

        self.formula = bAlways(and_all.evaluate)
        self.maxs = torch.tensor(MAXES_obs).to(device)
        self.mins = torch.tensor(MINS_obs).to(device)

    def forward(self, noise):

        signal = self.generator(noise, self.cond).to(device)

        rescaled_signal = self.mins.unsqueeze(1).unsqueeze(0)+((signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)

        return self.formula.evaluate(rescaled_signal).to(device)#[:,0] 


class condGen_w_QuantSTL_obs(nn.Module):
    def __init__(self, generator, cond):
        super(condGen_w_QuantSTL_obs, self).__init__()

        self.generator = generator
        self.cond = cond.to(device) 

        safety_1 = stl.Atom(var_index=2, threshold=R1_o, lte = False)
        safety_2 = stl.Atom(var_index=3, threshold=R2_o, lte = False)
        safety_3 = stl.Atom(var_index=4, threshold=R3_o, lte = False)
        safety_4 = stl.Atom(var_index=5, threshold=R4_o, lte = False)
        
        and_12 = stl.And(safety_1, safety_2)
        and_34 = stl.And(safety_3, safety_4)
        safe_and = stl.And(and_12, and_34)

        
        # for CROSSROAD Eventually(Globally (X< thresh))
        # x < 15 and 27 < y < 35
        atom_xu = stl.Atom(var_index=0, threshold=30, lte=True)
        atom_xl = stl.Atom(var_index=0, threshold=0, lte=False)
        atom_yu = stl.Atom(var_index=1, threshold=30, lte=True)
        atom_yl = stl.Atom(var_index=1, threshold=0, lte=False)

        and_x = stl.And(atom_xl, atom_xu)
        and_y = stl.And(atom_yl, atom_yu)
        and_xy = stl.And(and_x, and_y)

        and_all = stl.And(and_xy, safe_and)
        

        self.formula = glob = stl.Globally(and_all, unbound=True)
        self.maxs = torch.tensor(MAXES_obs).to(device)
        self.mins = torch.tensor(MINS_obs).to(device)  

    def signal_norm(self, x, center):

        return torch.norm(x-center.unsqueeze(2),np.inf, dim=1).unsqueeze(1).to(device)

    def forward(self, noise, return_traj = False):


        signal = self.generator(noise, self.cond).to(device)
        rescaled_signal = self.mins.unsqueeze(1).unsqueeze(0)+((signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)

        sign_obs1 = self.signal_norm(rescaled_signal,torch.tensor(C1_o,device=device))
        sign_obs2 = self.signal_norm(rescaled_signal,torch.tensor(C2_o,device=device))
        sign_obs3 = self.signal_norm(rescaled_signal,torch.tensor(C3_o,device=device))
        sign_obs4 = self.signal_norm(rescaled_signal,torch.tensor(C4_o,device=device))

        concat_signal = torch.cat((rescaled_signal, sign_obs1, sign_obs2, sign_obs3, sign_obs4), dim=1)

        if return_traj:
            return self.formula.quantitative(concat_signal), rescaled_signal
        else:
            return self.formula.quantitative(concat_signal)
        


#------------CROSSROAD--------------


MAXES_c = [50.77296056, 49.36994646]
MINS_c = [5.43242922, 0.06992657]

class condGen_crossroad(nn.Module):
    def __init__(self, generator, cond):
        super(condGen_crossroad, self).__init__()

        self.generator = generator
        self.cond = cond.to(device)
        self.maxs = torch.tensor(MAXES_c).to(device) 
        self.mins = torch.tensor(MINS_c).to(device)

    def forward(self, noise):

        signal = self.generator(noise, self.cond).to(device)
        #rescaled_signal = (self.mins+(signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)
        rescaled_signal = self.mins.unsqueeze(1).unsqueeze(0)+((signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)

        return rescaled_signal


class condGen_w_BoolSTL_crossroad(nn.Module):
    def __init__(self, generator, cond):
        super(condGen_w_BoolSTL_crossroad, self).__init__()

        self.generator = generator
        self.cond = cond.to(device)
        
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

        self.maxs = torch.tensor(MAXES_c).to(device) 
        self.mins = torch.tensor(MINS_c).to(device)
    
    def forward(self, noise):

        signal = self.generator(noise, self.cond).to(device)

        #rescaled_signal = (self.mins+(signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)
        rescaled_signal = self.mins.unsqueeze(1).unsqueeze(0)+((signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)

        return self.formula.evaluate(rescaled_signal).to(device) 


class condGen_w_QuantSTL_crossroad(nn.Module):
    def __init__(self, generator, cond):
        super(condGen_w_QuantSTL_crossroad, self).__init__()

        self.generator = generator
        self.cond = cond.to(device) 
        
        # for CROSSROAD Eventually(Globally (X< thresh))
        # x < 15 and 27 < y < 35
        atom_x = stl.Atom(var_index=0, threshold=15, lte=True)
        atom_yu = stl.Atom(var_index=1, threshold=35, lte=True)
        atom_yl = stl.Atom(var_index=1, threshold=27, lte=False)

        and_y = stl.And(atom_yl, atom_yu)
        and_xy = stl.And(and_y, atom_x)
        glob = stl.Globally(and_xy, unbound=True)

        self.formula = stl.Eventually(glob, unbound=True, time_bound=30)

        self.maxs = torch.tensor(MAXES_c).to(device) 
        self.mins = torch.tensor(MINS_c).to(device)
    
    def forward(self, noise, return_traj = False):

        signal = self.generator(noise, self.cond).to(device)

        #rescaled_signal = (self.mins+(signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)
        rescaled_signal = self.mins.unsqueeze(1).unsqueeze(0)+((signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)

        if return_traj:
            return self.formula.quantitative(rescaled_signal), rescaled_signal
        else:
            return self.formula.quantitative(rescaled_signal)
        


#---------------CITY---------------

MAXES = [219.9921619,  219.98706919,  99.99902845]
MINS = [0.0102229,  0.00505363, 4.45518785]

OBS1 = [[95.,150., 70.]]
OBS2a = [[25.,190., 55.]]
OBS2b = [[25.,165., 55.]]
OBS3 = [[75.,65., 80.]]
OBS4a = [[182.5, 25, 70.]]
OBS4b = [[182.5, 10., 70.]]
OBS5a = [[182.5,175., 90.]]
OBS5b = [[182.5,200., 90.]]
CENTER = [[100.,100.,100.]]
#R1, R2, R3, R4, R5 = 50., 35., 50., 40., 50.
R1, R2, R3, R4, R5 = 30., 15., 20., 10., 10.

RTOT = 200.

class condGen_city(nn.Module):
    def __init__(self, generator, cond):
        super(condGen_city, self).__init__()

        self.generator = generator
        self.cond = cond.to(device)
        self.maxs = torch.tensor(MAXES).to(device)
        self.mins = torch.tensor(MINS).to(device)

    def forward(self, noise):

        signal = self.generator(noise, self.cond).to(device)
        rescaled_signal = self.mins.unsqueeze(1).unsqueeze(0)+((signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)

        return rescaled_signal


class condGen_w_BoolSTL_city(nn.Module):
    def __init__(self, generator, cond):
        super(condGen_w_BoolSTL_city, self).__init__()

        self.generator = generator
        self.cond = cond.to(device)

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
                
        atom_1 = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(OBS1,device=device).unsqueeze(2), radius = torch.tensor(R1,device=device), lte=False)
        
        atom_2a = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(OBS2a,device=device).unsqueeze(2), radius = torch.tensor(R2,device=device), lte=False)
        atom_2b = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(OBS2b,device=device).unsqueeze(2), radius = torch.tensor(R2,device=device), lte=False)
        atom_3 = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(OBS3,device=device).unsqueeze(2), radius = torch.tensor(R3,device=device), lte=False)
        atom_4a = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(OBS4a,device=device).unsqueeze(2), radius = torch.tensor(R4,device=device), lte=False)
        atom_4b = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(OBS4b,device=device).unsqueeze(2), radius = torch.tensor(R4,device=device), lte=False)
        atom_5a = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(OBS5a,device=device).unsqueeze(2), radius = torch.tensor(R5,device=device), lte=False)
        atom_5b = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(OBS5b,device=device).unsqueeze(2), radius = torch.tensor(R5,device=device), lte=False)
        
        atom_2 = bAnd(atom_2a.evaluate, atom_2b.evaluate)
        atom_4 = bAnd(atom_4a.evaluate, atom_4b.evaluate)
        atom_5 = bAnd(atom_5a.evaluate, atom_5b.evaluate)

        atom_12 = bAnd(atom_1.evaluate, atom_2.evaluate)
        atom_123 = bAnd(atom_12.evaluate, atom_3.evaluate)
        atom_1234 = bAnd(atom_123.evaluate, atom_4.evaluate)
        atom_12345 = bAnd(atom_1234.evaluate, atom_5.evaluate)
        
        atom_bound = bAtomicPredicateNorm(atomic_predicate_norm, center=torch.tensor(CENTER,device=device).unsqueeze(2), radius = torch.tensor(RTOT,device=device), lte=True)
        
        atom_all = bAnd(atom_12345.evaluate, atom_bound.evaluate)
        self.formula = bAlways(atom_all.evaluate)
        #atom_all = bAnd(atom_1.evaluate, atom_bound.evaluate)
        #self.formula = bAlways(atom_all.evaluate)

        self.maxs = torch.tensor(MAXES).to(device) 
        self.mins = torch.tensor(MINS).to(device)

    def forward(self, noise):

        signal = self.generator(noise, self.cond).to(device)

        rescaled_signal = self.mins.unsqueeze(1).unsqueeze(0)+((signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)

        return self.formula.evaluate(rescaled_signal).to(device) 


class condGen_w_QuantSTL_city(nn.Module):
    def __init__(self, generator, cond):
        super(condGen_w_QuantSTL_city, self).__init__()

        self.generator = generator
        self.cond = cond.to(device) 
        
        # for CROSSROAD Eventually(Globally (X< thresh))

        
        atom_1 = stl.Atom(var_index=0, threshold=R1, lte=False)
        
        atom_2a = stl.Atom(var_index=1, threshold=R2, lte=False)
        atom_2b = stl.Atom(var_index=2, threshold=R2, lte=False)
        atom_3 = stl.Atom(var_index=3, threshold=R3, lte=False)
        atom_4a = stl.Atom(var_index=4, threshold=R4, lte=False)
        atom_4b = stl.Atom(var_index=5, threshold=R4, lte=False)
        atom_5a = stl.Atom(var_index=6, threshold=R5, lte=False)
        atom_5b = stl.Atom(var_index=7, threshold=R5, lte=False)
        
        atom_bound = stl.Atom(var_index=8, threshold=RTOT, lte=True)

        
        atom_2 = stl.And(atom_2a, atom_2b)
        atom_4 = stl.And(atom_4a, atom_4b)
        atom_5 = stl.And(atom_5a, atom_5b)

        atom_12 = stl.And(atom_1, atom_2)
        atom_123 = stl.And(atom_12, atom_3)
        atom_1234 = stl.And(atom_123, atom_4)
        atom_12345 = stl.And(atom_1234, atom_5)

        atom_all = stl.And(atom_12345, atom_bound)
        
        self.formula = stl.Globally(atom_all, unbound=True)
        
        #atom_all = stl.And(atom_1, atom_bound)
        
        #self.formula = stl.Globally(atom_all, unbound=True)
        self.maxs = torch.tensor(MAXES).to(device)  ### TODO: set the right values here
        self.mins = torch.tensor(MINS).to(device)

    def signal_norm(self, x, center):

        return torch.norm(x-center.unsqueeze(2),np.inf, dim=1).unsqueeze(1).to(device)

    def forward(self, noise, return_traj = False):

        signal = self.generator(noise, self.cond).to(device)
        rescaled_signal = self.mins.unsqueeze(1).unsqueeze(0)+((signal.permute(0,2,1)+1)*(self.maxs-self.mins)/2).permute(0,2,1)

        obs1_sign = self.signal_norm(rescaled_signal,torch.tensor(OBS1,device=device))
        obs2a_sign = self.signal_norm(rescaled_signal,torch.tensor(OBS2a,device=device))
        obs2b_sign = self.signal_norm(rescaled_signal,torch.tensor(OBS2b,device=device))
        
        obs3_sign = self.signal_norm(rescaled_signal,torch.tensor(OBS3,device=device))
        obs4a_sign = self.signal_norm(rescaled_signal,torch.tensor(OBS4a,device=device))
        obs4b_sign = self.signal_norm(rescaled_signal,torch.tensor(OBS4b,device=device))
        
        obs5a_sign = self.signal_norm(rescaled_signal,torch.tensor(OBS5a,device=device))
        obs5b_sign = self.signal_norm(rescaled_signal,torch.tensor(OBS5b,device=device))

        bound_sign =  self.signal_norm(rescaled_signal,torch.tensor(CENTER,device=device))
        concat_signal = torch.cat((obs1_sign, obs2a_sign, obs2b_sign, obs3_sign, obs4a_sign,obs4b_sign, obs5a_sign, obs5b_sign, bound_sign), dim=1)

        if return_traj:
            return self.formula.quantitative(concat_signal), rescaled_signal
        else:
            return self.formula.quantitative(concat_signal)