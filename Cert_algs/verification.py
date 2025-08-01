import numpy as np
import math
import torch
import os
from copy import copy
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import matplotlib.pyplot as plt
import pickle
plt.rcParams.update({'font.size': 22})

torch.manual_seed(5)

cuda = True if torch.cuda.is_available() else False
cuda = False
device = 'cuda' if cuda else 'cpu'
print('verification device = ', device)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def grad_ascent(qmodel, z0, lr= 0.005, tol = 10^(-6)):
  z0.requires_grad = True
  norm = np.inf
  z = copy(z0)
  z0 = torch.randn_like(z0)
  z0.requires_grad = True
  c = 0
  while(torch.norm(z-z0, norm)> tol and c < 1000):
    z0 = copy(z)
    z0.requires_grad = True
    robustness = qmodel(z)
    if (c+1)%100 == 0:
      print('current robustness is: ', robustness)

    gradient = torch.autograd.grad(robustness, z, allow_unused=True,retain_graph=True)[0]
    #print('current gradient is: ', gradient)
    z = z0 + lr * gradient
    c += 1
  print('------------- optimal robustness = ', robustness)
  
  return z0


    

def grad_ascent_correct(qmodel, z0, lr=0.005, tol=10**(-4), rec_iter=1, max_recursion=20):
    
    z0.requires_grad = True
    gradient = torch.ones_like(z0)
    c = 0
    print('Starting robustness is: ', qmodel(z0))
    
    while (torch.norm(gradient, p=float('inf')) > tol and c < 400):
        #z0 = z.clone()  # Update z0 to the current z
        z0.requires_grad = True  # Ensure z0 requires gradient
        
        robustness = qmodel(z0)  # Evaluate the robustness

        if (c + 1) % 50 == 0:
            print('Current robustness is: ', robustness)

        # Compute the gradient of robustness with respect to z
        gradient = torch.autograd.grad(robustness, z0, allow_unused=True)[0]

        # Ensure the gradient is not None before using it
        if gradient is not None:
          with torch.no_grad():
            z0 = z0 + lr * gradient  # Update z based on the gradient
        else:
            print("Gradient is None, stopping the optimization.")
            break
        
        
        c += 1
    
    print('------------- Optimal robustness = ', robustness)
    
    if qmodel(z0) > 0:
        return z0.detach()  # Return the optimized z
    else:
        print('No convergence, gradient ascent restarts after N iterations: ', rec_iter)
        
        if rec_iter >= max_recursion:
            print("Maximum recursion depth reached, stopping.")
            return None

        #lr2 = 0.005 + rec_iter * 0.0005  # Incrementally increase learning rate
        return grad_ascent_correct(qmodel, z0.detach(), lr=lr, tol=10**(-4), rec_iter=rec_iter + 1)
    

def grad_ascent_opt(qmodel, z0, lr=0.005, tol=10**(-4)):
    
    z0.requires_grad = True
    gradient = torch.ones_like(z0)
    c = 0
    print('Starting robustness is: ', qmodel(z0))
    
    while (torch.norm(gradient, p=float('inf')) > tol and c < 300):
        #z0 = z.clone()  # Update z0 to the current z
        z0.requires_grad = True  # Ensure z0 requires gradient
        
        robustness = qmodel(z0)  # Evaluate the robustness

        if (c + 1) % 50 == 0:
            print('Current robustness is: ', robustness)

        # Compute the gradient of robustness with respect to z
        gradient = torch.autograd.grad(robustness, z0, allow_unused=True)[0]

        # Ensure the gradient is not None before using it
        if gradient is not None:
          with torch.no_grad():
            z0 = z0 + lr * gradient  # Update z based on the gradient
        else:
            print("Gradient is None, stopping the optimization.")
            break
        
        
        c += 1
    
    print('------------- Optimal robustness = ', robustness)
    
    return z0.detach()  # Return the optimized z
  



def traj_verifier(cmodel, zopt, epsilon, latent_dim = 50, maxes = None):

  
  norm = np.inf
  ptb = PerturbationLpNorm(norm = norm, eps = epsilon)

  bounded_model_bstl = BoundedModule(cmodel, torch.zeros_like(zopt), bound_opts={"conv_mode": "tensor"}, verbose=False)
  bounded_model_bstl.eval()
  bounded_z = BoundedTensor(zopt, ptb)

  with torch.no_grad():  # If gradients of the bounds are not needed, we can use no_grad to save memory.
    lb, ub = bounded_model_bstl.compute_bounds(x=(bounded_z), method='CROWN')
  
  
  rescaled_lb = ((lb.permute(0,2,1)+1)*maxes/2).permute(0,2,1)
  rescaled_ub = ((ub.permute(0,2,1)+1)*maxes/2).permute(0,2,1)

  print(f'lb = {rescaled_lb}, ub = {rescaled_ub}')


def verifier(bmodel, qmodel, M=1, epsilon = 0.0001, latent_dim = 50):

  B_list = [] # list of (z,eps, lb, ub) pairs
  norm = np.inf
  ptb = PerturbationLpNorm(norm = norm, eps = epsilon)
  for i in range(M):
    z0_i = Tensor(np.random.normal(0, 1, (1, latent_dim)))
    zstar_i = grad_ascent(qmodel, z0_i)
    print('z pivot = ', zstar_i)
    bounded_model_bstl = BoundedModule(bmodel, zstar_i, bound_opts={"conv_mode": "tensor"}, verbose=True)
    bounded_model_bstl.eval()
    bounded_z = BoundedTensor(zstar_i, ptb)

    with torch.no_grad():  # If gradients of the bounds are not needed, we can use no_grad to save memory.
      lb_bstl, ub_bstl = bounded_model_bstl.compute_bounds(x=(bounded_z), method='CROWN')
    
    print(f'lb = {lb_bstl}, ub = {ub_bstl}')
    if lb_bstl == 1 and len(B_list) == 0:
      flag = 1
    elif lb_bstl == 1 and len(B_list) > 0:
      flag = 1
      for l in range(len(B_list)):
        if torch.norm(B_list[l][0]-zstar_i, norm) >= B_list[l][1]+epsilon:
          flag = 0
    else:
      flag = 0

    if flag == 1:
      B_list.append((zstar_i, epsilon, lb_bstl, ub_bstl))
    
  return B_list

def compute_induced_prob(B_list, latent_dim):
  prob = 0
  norm = np.inf
  m = len(B_list)
  if m > 0:
    for i in range(m):
      ball_i = B_list[i]
      zstar_i = ball_i[0]
      eps_i = ball_i[1]
      prod = 1
      for k in range(latent_dim):

        lbk = (zstar_i-eps_i)[0,k]
        ubk = (zstar_i+eps_i)[0,k]
        Ak = torch.special.erf((zstar_i[0,k]-lbk)/(eps_i*math.sqrt(2)/3))
        Bk = torch.special.erf((zstar_i[0,k]-ubk)/(eps_i*math.sqrt(2)/3))
        
        prod *= Ak-Bk
      print(f'prod {i+1}/{m}= {prod}')
      prob += prod/(2**latent_dim)
  return prob

def compute_prob(B_list, latent_dim):
  prob = 0
  norm = np.inf
  m = len(B_list)
  if m > 0:
    for i in range(m):
      ball_i = B_list[i]
      prod = 1
      for k in range(latent_dim):
        lbk = (ball_i[0]-ball_i[1])[0,k]
        ubk = (ball_i[0]+ball_i[1])[0,k]
        Ak = torch.special.erf(-lbk/math.sqrt(2))
        Bk = torch.special.erf(-ubk/math.sqrt(2))
        
        prod *= Ak-Bk
      print(f'prod {i+1}/{m}= {prod}')
      prob += prod/(2**latent_dim)
  return prob


def verifier_increment(bmodel, qmodel, M=1, epsilon = 0.0005, delta_eps = 0.0005, latent_shape = Tensor(np.empty((1,50)))):

  B_list = [] # list of (z,eps, lb, ub) pairs
  norm = np.inf
  eps_start = epsilon

  for i in range(M):
    print(f'i = {i+1}/{M}')
    ver_flag = 0
    #z0_i = Tensor(np.random.normal(0, 1, (1, latent_dim)))
    z0_i = torch.randn_like(latent_shape, device = device)
    print('center shape is: ', z0_i.shape)
    zstar_i = grad_ascent(qmodel, z0_i).to(device)
    #bmodel = bmodel.to(device)
    print('z pivot = ', zstar_i)
    #print('model device is. ', next(bmodel.parameters()).is_cuda)
    zstar = zstar_i.detach()
 
    epsilon = eps_start


    for l in range(len(B_list)):
      if torch.norm(B_list[l][0]-zstar_i, norm) <= B_list[l][1]+epsilon:
        loc_flag = 0

    bounded_model_bstl = BoundedModule(bmodel, torch.zeros_like(latent_shape), bound_opts={"conv_mode": "tensor",'sparse_intermediate_bounds': False,
            'sparse_conv_intermediate_bounds': False,'sparse_intermediate_bounds_with_ibp': False},verbose=False)
    bounded_model_bstl.eval()

    loc_flag = 1
    while(loc_flag == 1):
      print(f'eps = {epsilon}')
      ptb = PerturbationLpNorm(norm = norm, eps = epsilon)
      bounded_z = BoundedTensor(zstar, ptb)


      with torch.no_grad():  # If gradients of the bounds are not needed, we can use no_grad to save memory.
        lb_bstl, ub_bstl = bounded_model_bstl.compute_bounds(x=(bounded_z), method='backward')
        print('lower bound is: ', lb_bstl)
        print('upper bound is: ', ub_bstl)




      if lb_bstl == 1 and len(B_list) == 0:
        loc_flag = 1
      elif lb_bstl == 1 and len(B_list) > 0:
        loc_flag = 1
        for l in range(len(B_list)):
          if torch.norm(B_list[l][0]-zstar_i, norm) <= B_list[l][1]+epsilon:
            loc_flag = 0
      else:
        loc_flag = 0

      if loc_flag == 1:
        ver_flag = 1
        last_ball = (zstar_i, epsilon, lb_bstl, ub_bstl)
        epsilon += delta_eps


    if ver_flag == 1:
      B_list.append(last_ball)


  return B_list

def verifier_increment2(bmodel, qmodel, M=1, eps_start = 0.001, delta_eps = 0.0005, latent_shape = np.empty((1,50)), model_id=''):

  B_list = [] # list of (z,eps, lb, ub) pairs
  norm = np.inf
  discard_count=0
  n_disc=0

  for i in range(M):
    print(f'i = {i+1}/{M}')
    ver_flag = 0
    #z0_i = Tensor(np.random.normal(0, 1, (1, latent_dim)))
    
    rob = -100
    it = 1
    while(rob<0):
      z0_i = torch.randn_like(latent_shape, device = device)
      print('center shape is: ', z0_i.shape)
      Zstar = grad_ascent_opt(qmodel, z0_i).to(device)
      rob = qmodel(Zstar)
      print('iterations of Gradient Ascent number: ', it)
    #zstar_i = torch_grad(qmodel, z0_i).to(device)
    #bmodel = bmodel.to(device)
    print('z pivot = ', Zstar)



    loc_flag = 1
    for l in range(len(B_list)):
      #if torch.norm(B_list[l][0]-Zstar, norm) <= B_list[l][1]+epsilon:
      if torch.norm(B_list[l][0]-Zstar, norm) <= 0.001:
        loc_flag = 0
        discard_count+=1
        print('The starting point was too close to an exisiting ball! Restarting the search')
    epsilon = eps_start
    bounded_model = BoundedModule(bmodel, torch.zeros_like(Zstar), bound_opts={"conv_mode": "tensor",'sparse_intermediate_bounds': False,
            'sparse_conv_intermediate_bounds': False,'sparse_intermediate_bounds_with_ibp': False},verbose=True)
    bounded_model.eval()

    bound_count=0
    while(loc_flag == 1):
      eps = epsilon
      norm = norm
      ptb = PerturbationLpNorm(norm = norm, eps = eps)
      # Input tensor is wrapped in a BoundedTensor object.
      bounded_traj_z = BoundedTensor(Zstar, ptb)
      print('Model prediction:', bounded_model(bounded_traj_z))

      print('Bounding method: backward (CROWN, DeepPoly)')
      with torch.no_grad():  # If gradients of the bounds are not needed, we can use no_grad to save memory.
        lb_bstl, ub_bstl = bounded_model.compute_bounds(x=(bounded_traj_z), method='backward')
      bound_count+=1
      computed_ball = (Zstar, eps, lb_bstl, ub_bstl)
      #ball_res_path = os.path.join('results','DIFF',model_id, 'HOM','HomBall_interm_res_'+ model_id+'_'+ str(i) +'.pkl')
      #ball_res_path = 'Ball_res_'+ model_id+'_'+ str(i) +'.pkl'
      #with open(ball_res_path, "wb") as f:
      #      pickle.dump(last_ball, f)
      if lb_bstl != 1:
        print('lower bound does not satisfy the condition, ball discarded')
        print(computed_ball)
        n_disc+=1

      if lb_bstl == 1 and len(B_list) == 0:
        loc_flag = 1
      elif lb_bstl == 1 and len(B_list) > 0:
        loc_flag = 1
        for l in range(len(B_list)):
          if torch.norm(B_list[l][0]-Zstar, norm) <= B_list[l][1]+epsilon:
            loc_flag = 0
      else:
        loc_flag = 0

      if loc_flag == 1:
        ver_flag = 1
        last_ball = (Zstar, eps, lb_bstl, ub_bstl)
        epsilon += delta_eps


    if ver_flag == 1:
      B_list.append(last_ball)
      print(last_ball)

      list_path = os.path.join('results','DIFF',model_id, 'HOM', model_id+'_HOM_small_eps_N='+ str(len(B_list))+'.pkl')
      tmp_balls = {"ball_list": B_list}
      with open(list_path, "wb") as f:
          pickle.dump(tmp_balls, f)
  print('Procedure ended with a total number of bound computation', bound_count)
  print('Procedure ended with a list of length: ', len(B_list))
  print('Due to their closeness, the procedure ended up discarding balls N= ', discard_count)
  print('The number of discarded balls was:', n_disc)
  print(B_list)

  return B_list  



def verifier_heterog_increment2(bmodel, qmodel,weight = 0.1, M=1, epsilon = 0.0005, delta_eps = 0.0005, latent_shape = Tensor(np.empty((1,50))), model_id=''):


  B_list = [] # list of (z,eps, lb, ub) pairs
  norm = np.inf

  for i in range(M):
    print(f'i = {i+1}/{M}')
    ver_flag = 0
    #z0_i = Tensor(np.random.normal(0, 1, (1, latent_dim)))
    z0_i = torch.randn_like(latent_shape, device = device)
    print('center shape is: ', z0_i.shape)
    zstar_i = grad_ascent(qmodel, z0_i).to(device)
    #bmodel = bmodel.to(device)
    print('z pivot = ', zstar_i)
    print('model device is. ', next(bmodel.parameters()).is_cuda)

    print('shape of z_star is ',zstar_i.shape)
    print('type of z_star is ',type(zstar_i))

    
    print('The parameters of the model are on cuda? ',next(bmodel.parameters()).is_cuda)

    Zstar = zstar_i.detach()

    loc_flag = 1
    eps_vec = epsilon*torch.ones_like(latent_shape)
    for l in range(len(B_list)):
      if torch.any(torch.abs(B_list[l][0]-zstar_i) - B_list[l][1]+eps_vec <= 0 ):
        loc_flag = 0

   

    bounded_model = BoundedModule(bmodel, torch.zeros_like(Zstar), bound_opts={"conv_mode": "tensor",'sparse_intermediate_bounds': False,
            'sparse_conv_intermediate_bounds': False,'sparse_intermediate_bounds_with_ibp': False},verbose=True)
    bounded_model.eval()

    bound_count=0
    while(loc_flag == 1):

      Zstar = zstar_i.detach()

      eps = epsilon
      norm = norm
      ptb = PerturbationLpNorm(norm = norm, eps = eps)
      # Input tensor is wrapped in a BoundedTensor object.
      bounded_traj_z = BoundedTensor(Zstar, ptb)
      print('Model prediction:', bounded_model(bounded_traj_z))

      print('Bounding method: backward (CROWN, DeepPoly)')
      #loc_flag = 1
      with torch.no_grad():  # If gradients of the bounds are not needed, we can use no_grad to save memory.
        lb_bstl, ub_bstl = bounded_model.compute_bounds(x=(bounded_traj_z), method='backward')
      bound_count+=1
      last_ball = (Zstar, eps_vec, lb_bstl, ub_bstl)
      ball_res_path = 'HetBall_interm_res_'+ model_id+'_'+ str(i) +'_w_'+ str(bound_count)+'.pkl'
      with open(ball_res_path, "wb") as f:
            pickle.dump(last_ball, f)

      if lb_bstl == 1 and len(B_list) == 0:
        loc_flag = 1
      elif lb_bstl == 1 and len(B_list) > 0:
        loc_flag = 1
        for l in range(len(B_list)):
          #if torch.norm(B_list[l][0]-zstar_i, norm) <= B_list[l][1]+eps_vec:
          if torch.any(torch.abs(B_list[l][0]-zstar_i) - B_list[l][1]+eps_vec <= 0 ):
            loc_flag = 0
      else:
        loc_flag = 0

      if loc_flag == 1:
        ver_flag = 1
        last_ball = (zstar_i, eps_vec, lb_bstl, ub_bstl)
        
        robustness = qmodel(zstar_i)
        gradient = torch.autograd.grad(robustness, zstar_i, allow_unused=True,retain_graph=True)[0]
        grad_increm = (weight/2)*gradient/torch.norm(gradient, norm)
        print('GRAD INCREMENT = ', grad_increm)
        zstar_i = zstar_i + grad_increm
        eps_vec = eps_vec + torch.abs(grad_increm)


    if ver_flag == 1:
      B_list.append(last_ball)

      list_path = os.path.join('results','DIFF', model_id+'_HET_N='+str(len(B_list))+'.pkl')
      tmp_balls = {"ball_list": B_list}
      with open(list_path, "wb") as f:
          pickle.dump(tmp_balls, f)


  return B_list



from scipy.stats import truncnorm




def prob_vec(logprobs):
  realprobs=[]
  for i in range(len(logprobs)):
    sumvec=0
    for j in range(len(logprobs)):
      sumvec+=np.exp(logprobs[j] -logprobs[i])
    realprobs.append(1/sumvec)
  realprobs = np.array(realprobs)
  realprobs = realprobs.astype('float64')
  realprobs = realprobs+ 0.05
  realprobs = realprobs/realprobs.sum()
  return(realprobs)






def compute_log_prob(B_list, latent_dim):
  m = len(B_list)
  logprob = torch.empty(m, device=device)
  print('length of ball list is: ',m)
  if m > 0:
    for i in range(m):
      ball_i = B_list[i]
      logprod = 1
      for k in range(latent_dim):
        lbk = (ball_i[0]-ball_i[1])[0,:,k]
        ubk = (ball_i[0]+ball_i[1])[0,:,k]
        Ak = torch.special.erf(-lbk/math.sqrt(2))
        Bk = torch.special.erf(-ubk/math.sqrt(2))
        
        logprod += torch.log((Ak-Bk)/2).sum()
        print(logprod)
      
      logprob[i] = logprod
    #print('1. logprob before nan removal = ',logprob )
    logprob[logprob == -float('inf')] = 0
    logprob = torch.nan_to_num(logprob, nan=0)
    #print(logprob == -float('inf'))
    
    #print('2. logprob after nan removal = ',logprob )


  p_vec = prob_vec(logprob.detach().cpu().numpy())

  #print('p_vec sum = ', p_vec.sum())



  #if p_vec.sum() != 1:
  #  print('WARNING: probabiliries do not sum up to one!!! Changed to uniform.')
  #  p_vec = np.ones(m)/m

  return logprob, p_vec



def truncated_samples(ball_i, latent_dim, nsamples):
    #print('eps_vec = ', ball_i[1])
    #ball_i[1][ball_i[1] < 0] = 0
    LB = (ball_i[0]-(ball_i[1]-0.001)).detach().cpu().numpy()
    UB = (ball_i[0]+(ball_i[1]-0.001)).detach().cpu().numpy()
    Z = np.empty((nsamples, latent_dim))
    for k in range(latent_dim):
        lb_ik = LB[0,k]
        ub_ik = UB[0,k]

        #print('lb_ik = ', lb_ik)
        #print('ub_ik = ', ub_ik)
       
        Z[:,k] = truncnorm.rvs(lb_ik, ub_ik, loc= 0, scale=1, size=nsamples)
    
    return torch.tensor(Z).to(device)

def truncated_samples_tensor(ball_i, latent_dim, latent_shape, nsamples):
    LB = (ball_i[0]-ball_i[1]).detach().cpu().numpy()
    UB = (ball_i[0]+ball_i[1]).detach().cpu().numpy()
    Z_single = torch.empty_like(latent_shape)
    #print('Z_single shape is: ', Z_single.shape)
    Z = Z_single.repeat(nsamples,1,1)
    #Z = Z.permute(0,2,1)
    print('Z shape now is: ', Z.shape)
    Z = Z.detach().cpu().numpy()
    
    for k in range(latent_dim):
        for i in range(Z.shape[1]):
          lb_ik = LB[0,i,k]
          ub_ik = UB[0,i,k]
          Z[:,i,k] = truncnorm.rvs(lb_ik, ub_ik, loc= 0, scale=1, size=nsamples)

       
        #Z[:,:,k] = truncnorm.rvs(lb_ik, ub_ik, loc= 0, scale=1, size=nsamples)
    Z_final = torch.tensor(Z).to(device)
    print('Shape of generated truncated tensor is: ', Z_final.shape)
    return Z_final

import time

def empirical_satisfaction(ball_list, qmodel, latent_dim, latent_shape, nsamples = 100, model_name = ''):

  M = len(ball_list)
  _, ball_probs = compute_log_prob(ball_list, latent_dim)
  print('probability per ball = ', ball_probs)
  samples_per_ball = np.random.choice(np.arange(M), nsamples, p=ball_probs)

  Z_single = torch.empty_like(latent_shape)
    #print('Z_single shape is: ', Z_single.shape)
  Z = Z_single.repeat(nsamples,1,1)
  c=0
  
  start = time.time()
  for i in range(M):
    ball_i = ball_list[i]
    nsample_i = np.count_nonzero(samples_per_ball==i)

    Zi = truncated_samples_tensor(ball_i, latent_dim,latent_shape, nsample_i)
    
    Z[c:c+nsample_i] = Zi
    c += nsample_i

  fig = plt.figure()
  sat = torch.empty(nsamples, device=device)
  for j in range(nsamples):
    rob_j, traj_j = qmodel(Z[j:j+1], return_traj=True)

    if traj_j.shape[1] == 2:
      traj_jj = traj_j.detach().cpu().numpy()
      plt.plot(traj_jj[0,0], traj_jj[0,1], c='g')
    
    sat[j] = torch.sign(torch.relu(rob_j))
  plt.title(model_name)
  plt.tight_layout()
  fig_path = os.path.join('results','DIFF',model_name, 'EXP','diff_'+model_name+'_certified_samples_'+str(nsamples)+'.png')
  fig.savefig(fig_path)
  plt.close()

  print(f'Empirical satisfaction over {nsamples} samples = ', sat.mean())
  print(f'Time to compute {nsamples} is ', time.time()-start)
  return Z, sat

