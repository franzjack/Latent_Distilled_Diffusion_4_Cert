import os
import sys
import math
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm
sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader
#import torch_two_sample 

from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

from Dataset_Cross import *

from critic_start import *
from Models.GAN.generator import *
#from stl_utils import *
from model_details import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=4000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--gen_lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--crit_lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=14, help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=10, help="number of training steps for discriminator per iter")
parser.add_argument("--traj_len", type=int, default=14, help="number of steps")
parser.add_argument("--n_test_trajs", type=int, default=1, help="number of trajectories per point at test time")
parser.add_argument("--x_dim", type=int, default=2, help="number of channels of x")
parser.add_argument("--y_dim", type=int, default=2, help="number of channels of y")
parser.add_argument("--model_name", type=str, default="OBS", help="name of the model")
parser.add_argument("--species_labels", type=str, default=[], help="list of species names")
parser.add_argument("--training_flag", type=eval, default=True, help="do training or not")
parser.add_argument("--loading_id", type=str, default="", help="id of the model to load")
parser.add_argument("--active_flag", type=eval, default=False)
parser.add_argument("--Q", type=int, default=90, help="threshold quantile for the active query strategy")
parser.add_argument("--rob_flag", type=eval, default=False)
parser.add_argument("--finetune_flag", type=eval, default=True)
parser.add_argument("--optimizer", type=str, default='RAdam')
parser.add_argument("--map_type", type=str, default="obs")

opt = parser.parse_args()
print(opt)

#opt = get_model_details(opt)

opt.y_dim = opt.x_dim
opt.latent_dim = opt.traj_len
cuda = True if torch.cuda.is_available() else False
#cuda = False
model_name = opt.model_name
if opt.active_flag:
    if opt.model_name == 'TS':
        trainset_fn = 'GAN/Drive_WGAN/'+ model_name+f"_wgan{opt.loading_id}_{opt.Q}perc_retrain_set_H={opt.traj_len}_{4000}x10.pickle"
    else:
        if not opt.finetune_flag:
            trainset_fn = "GAN/Drive_WGAN/"+model_name+f"_wgan{opt.loading_id}_{opt.Q}perc_retrain_set_H={opt.traj_len}_{int(2000+(100-opt.Q)*20)}x10.pickle"
        else:
            trainset_fn = "GAN/Drive_WGAN/"+model_name+f"_wgan{opt.loading_id}_{opt.Q}perc_retrain_set_H={opt.traj_len}_{int((100-opt.Q)*20)}x10.pickle"


    testset_fn = "GAN/Drive_WGAN/"+model_name+f"_test_set_H={opt.traj_len}.pickle"
    validset_fn = "GAN/Drive_WGAN/"+model_name+f"_valid_set_H={opt.traj_len}.pickle"
else:

    trainset_fn = 'data/'+opt.map_type +'/'+opt.map_type +'_map_data_train.pickle'
    testset_fn = 'data/'+opt.map_type +'/'+opt.map_type +'_map_data_test.pickle'
    validset_fn = 'data/'+opt.map_type +'/'+opt.map_type +'_map_data_test.pickle'


ds = Dataset(trainset_fn, testset_fn, opt.x_dim, opt.y_dim, opt.traj_len)
ds.add_valid_data(validset_fn)
ds.load_train_data()

print('max and min', ds.HMAX, ds.HMIN)


if opt.training_flag and not opt.active_flag:
    ID = str(np.random.randint(0,500))
    #ID = str(10)
    print("ID = ", ID)
    plots_path = "save/"+model_name+"/GAN/ID_"+ID
    parent_path = plots_path
elif opt.active_flag:
    #ID = 'Retrain_'+opt.loading_id+f'_{opt.Q}perc'
    parent_path = "save/"+model_name+f'/GAN/ID_{opt.loading_id}'
    if opt.finetune_flag:
        plots_path = parent_path+f'/FineTune_{opt.n_epochs}ep_lr={opt.gen_lr}_{opt.Q}perc'
    else:
        plots_path = parent_path+f'/Retrain_{opt.n_epochs}ep_lr={opt.gen_lr}_{opt.Q}perc'

else:
    ID = opt.loading_id
    plots_path = "save/"+model_name+"/GAN/ID_"+ID
    parent_path = plots_path


os.makedirs(plots_path, exist_ok=True)
f = open(plots_path+"/log.txt", "w")
f.write(str(opt))
f.close()

GEN_PATH = plots_path+"/generator.pt"
CRIT_PATH = plots_path+"/critic.pt"
# Loss weight for gradient penalty
lambda_gp = 10


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(C, real_samples, fake_samples, lab):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    c_interpolates = C(interpolates, lab)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=c_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    gradients = gradients.reshape(gradients.shape[0], opt.traj_len*opt.x_dim)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def generate_random_conditions():
    return (np.random.rand(opt.batch_size, opt.y_dim, 1)-0.5)*2

# ----------
#  Training
# ----------
# Initialize generator and critic


if opt.training_flag:

    st = time.time()

    if opt.active_flag and opt.finetune_flag:
        critic = torch.load(parent_path+"/critic.pt")
        generator = torch.load(parent_path+"/generator.pt")
        critic.train()
        generator.train()
    else:
        generator = Generator(opt.x_dim, opt.traj_len, opt.latent_dim)
        critic = Critic(opt.x_dim, opt.traj_len)

    if cuda:
        generator.cuda()
        critic.cuda()
    
    # Optimizers
    if opt.optimizer == 'RAdam': 
        optimizer_G = torch.optim.RAdam(generator.parameters(), lr=opt.gen_lr, betas=(opt.b1, opt.b2))
        optimizer_C = torch.optim.RAdam(critic.parameters(), lr=opt.crit_lr, betas=(opt.b1, opt.b2))

    elif opt.optimizer == 'RMSP':
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.gen_lr)
        optimizer_C = torch.optim.RMSprop(critic.parameters(), lr=opt.crit_lr)




    
    batches_done = 0
    G_losses = []
    C_losses = []
    real_comp = []
    gen_comp = []
    gp_comp = []

    full_G_loss = []
    full_C_loss = []
    for epoch in range(opt.n_epochs):
        bat_per_epo = int(ds.n_points_dataset / opt.batch_size)
        n_steps = bat_per_epo * opt.n_epochs
        
        tmp_G_loss = []
        tmp_C_loss = []

        
        for i in range(bat_per_epo):
            trajs_np, conds_np = ds.generate_mini_batches(opt.batch_size)
            # Configure input
            real_trajs = Variable(Tensor(trajs_np))
            conds = Variable(Tensor(conds_np))
            

            #print('shape of conds is',conds.shape)
            # ---------------------
            #  Train Critic
            # ---------------------

            optimizer_C.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

            #print('noise shape:', z.shape)
            #print('cond shape is:', conds.shape)

            # Generate a batch of images
            fake_trajs = generator(z, conds)
            
            
            # Real images
            #print('conds shape is', conds.shape)
           
            #print('real_trajs shape is', real_trajs.shape)
            real_validity = critic(real_trajs, conds)
            # Fake images
            fake_validity = critic(fake_trajs, conds)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(critic, real_trajs.data, fake_trajs.data, conds.data)
            # Adversarial loss
            c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            real_comp.append(torch.mean(real_validity).item())
            gen_comp.append(torch.mean(fake_validity).item())
            gp_comp.append(lambda_gp * gradient_penalty.item())
            tmp_C_loss.append(c_loss.item())
            full_C_loss.append(c_loss.item())

            c_loss.backward(retain_graph=True)
            optimizer_C.step()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                gen_conds = Variable(Tensor(generate_random_conditions()))
                

                #print("input shape is",gen_conds.shape, "\n")
                #print("conditioning shape is ",z.shape, "\n")

                # Generate a batch of images
                gen_trajs = generator(z, gen_conds)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = critic(gen_trajs, gen_conds)
                g_loss = -torch.mean(fake_validity)
                tmp_G_loss.append(g_loss.item())
                full_G_loss.append(g_loss.item())
                g_loss.backward(retain_graph=True)
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [C loss: %f] [G loss: %f]"
                    % (epoch+1, opt.n_epochs, i, bat_per_epo, c_loss.item(), g_loss.item())
                )

                batches_done += opt.n_critic
        if (epoch+1) % 500 == 0:
            torch.save(generator, plots_path+"/"+str(epoch)+"_generator.pt")    
        C_losses.append(np.mean(tmp_C_loss))
        G_losses.append(np.mean(tmp_G_loss))
    
    training_time = time.time()-st
    print("WGAN Training time: ", training_time)
    print('The ID of the model is:',ID)
    f = open(plots_path+"/log.txt", "w")
    f.write("WGAN Training time: ")
    f.write(str(training_time))
    f.close()
    fig, axs = plt.subplots(2,1,figsize = (12,6))
    axs[0].plot(np.arange(opt.n_epochs), G_losses)
    axs[1].plot(np.arange(opt.n_epochs), C_losses)
    axs[0].set_title("generator loss")
    axs[1].set_title("critic loss")
    plt.tight_layout()
    fig.savefig(plots_path+"/losses.png")
    plt.close()

    fig1, axs1 = plt.subplots(2,1,figsize = (12,6))
    axs1[0].plot(np.arange(len(full_G_loss)), full_G_loss)
    axs1[1].plot(np.arange(len(full_C_loss)), full_C_loss)
    axs1[0].set_title("generator loss")
    axs1[1].set_title("critic loss")
    plt.tight_layout()
    fig1.savefig(plots_path+"/full_losses.png")
    plt.close()

    fig2, axs2 = plt.subplots(3,1, figsize = (12,9))
    axs2[0].plot(np.arange(n_steps), real_comp)
    axs2[1].plot(np.arange(n_steps), gen_comp)
    axs2[2].plot(np.arange(n_steps), gp_comp)
    axs2[0].set_title("real term")
    axs2[1].set_title("generated term")
    axs2[2].set_title("gradient penalty term")
    plt.tight_layout()
    fig2.savefig(plots_path+"/components.png")
    plt.close()

    # save the ultimate trained generator    
    torch.save(generator, GEN_PATH)
    torch.save(critic, CRIT_PATH)
else:
    # load the ultimate trained generator
    print("MODEL_PATH: ", GEN_PATH)
    generator = torch.load(GEN_PATH)
    generator.eval()
    if cuda:
        generator.cuda()

ds.load_test_data(opt.n_test_trajs)
#ds.load_valid_data()


TEST_TRAJ_FLAG = opt.training_flag
VALID_TRAJ_FLAG = False
TEST_PLOT_FLAG = True

HIST_FLAG = False
WASS_FLAG = False
STAT_TEST = False
TEST_STL_FLAG = False
VALID_STL_FLAG = False


if TEST_TRAJ_FLAG:
    print("Computing test trajectories...")
    n_gen_trajs = ds.n_test_traj_per_point
    gen_trajectories = np.empty(shape=(ds.n_points_test, n_gen_trajs, opt.x_dim, opt.traj_len))
    for iii in range(ds.n_points_test):
        #print("Test point nb ", iii+1, " / ", ds.n_points_test)
        for jjj in range(n_gen_trajs):
            st = time.time()
            z_noise = np.random.normal(0, 1, (1, opt.latent_dim))
            #print(ds.Y_test_transp[iii,jjj])
            temp_out = generator(Variable(Tensor(z_noise)), Variable(Tensor([ds.Y_test_transp[iii,jjj]])))
            #print('WGAN time to generate one traj: ', time.time()-st)
            gen_trajectories[iii,jjj] = temp_out.detach().cpu().numpy()[0]

    trajs_dict = {"gen_trajectories": gen_trajectories}
    file = open(plots_path+'/generated_test_trajectories.pickle', 'wb')
    # dump information to that file
    pickle.dump(trajs_dict, file)
    # close the file
    file.close()
else:
    file = open(plots_path+'/generated_test_trajectories.pickle', 'rb')
    trajs_dict = pickle.load(file)
    file.close()
    gen_trajectories = trajs_dict["gen_trajectories"]


if VALID_TRAJ_FLAG:
    print("Computing valid trajectories...")
    n_gen_trajs = ds.n_valid_traj_per_point
    gen_valid_trajectories = np.empty(shape=(ds.n_points_valid, n_gen_trajs, opt.x_dim, opt.traj_len))
    for iii in range(ds.n_points_valid):
        #print("Valid int nb ", iii+1, " / ", ds.n_points_valid)
        for jjj in range(n_gen_trajs):
            z_noise = np.random.normal(0, 1, (1, opt.latent_dim))
            #print(ds.Y_test_transp[iii,jjj])
            temp_out = generator(Variable(Tensor(z_noise)), Variable(Tensor([ds.Y_valid_transp[iii,jjj]])))
            gen_valid_trajectories[iii,jjj] = temp_out.detach().cpu().numpy()[0]

    valid_trajs_dict = {"gen_trajectories": gen_valid_trajectories}
    file = open(plots_path+'/generated_valid_trajectories.pickle', 'wb')
    # dump information to that file
    pickle.dump(valid_trajs_dict, file)
    # close the file
    file.close()
else:
    sss = 3
    #file = open(plots_path+'/generated_valid_trajectories.pickle', 'rb')
    #valid_trajs_dict = pickle.load(file)
    #file.close()
    #gen_valid_trajectories = valid_trajs_dict["gen_trajectories"]

#colors = ['blue', 'orange']
#leg = ['real', 'gen']

#PLOT TRAJECTORIES
if opt.x_dim == 2:
    plt.rcParams.update({'font.size': 25})
    print(gen_trajectories[0].shape)
    print(ds.HMAX)
    print(ds.HMIN)

    n_trajs_to_plot = 10
    print("Plotting test trajectories...")      
    tspan = range(opt.traj_len)
    for kkk in range(0, ds.n_points_test, int(ds.n_points_test/50)):
        #print("Test point nb ", kkk+1, " / ", ds.n_points_test)
        #fig, axs = plt.subplots(opt.x_dim,figsize=(16.0, opt.x_dim*4))
        G = np.array([(ds.HMIN+(gen_trajectories[kkk, it].T+1)*(ds.HMAX-ds.HMIN)/2).T for it in range(ds.n_test_traj_per_point)])
        pstart = np.array([(ds.HMIN+(ds.Y_test_transp[kkk,it].T+1)*(ds.HMAX-ds.HMIN)/2).T for it in range(ds.n_test_traj_per_point)])
        
        
        R = np.array([(ds.HMIN+(ds.X_test_transp[kkk, it].T+1)*(ds.HMAX-ds.HMIN)/2).T for it in range(ds.n_test_traj_per_point)])
        #G = (gen_trajectories[kkk].transpose((0,2,1))*ds.test_std+ds.test_mean).transpose((0,2,1))
        #R = ds.X_test_count[kkk].transpose((0,2,1))
        if kkk==0:
            print(pstart)
            
            print(pstart.shape)
            
            #print(plt.plot(G[:,0,:], G[:,1,:]))
            #plt.show()
            #print(G.shape)
            


        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.plot(G[:,0,:].T, G[:,1,:].T,'orange')
        ax.scatter(pstart[:,0,0], pstart[:,1,0],c='b')
        

        ax.plot(R[:,0,:].T, R[:,1,:].T,'blue')
        



        fig.suptitle('cwgan',fontsize=40)
        plt.tight_layout()
        fig.savefig(plots_path+"/WGAN_"+opt.model_name+"_stoch_rescaled_trajectories_point_"+str(kkk)+".png")
        plt.close()

#PLOT 3D_LINE

if opt.x_dim ==3:
    plt.rcParams.update({'font.size': 25})
    print(gen_trajectories[0].shape)
    print(ds.HMAX)
    print(ds.HMIN)

    n_trajs_to_plot = 10
    print("Plotting test trajectories...")      
    tspan = range(opt.traj_len)
    for kkk in range(0, ds.n_points_test, int(ds.n_points_test/50)):
        G = np.array([(ds.HMIN+(gen_trajectories[kkk, it].T+1)*(ds.HMAX-ds.HMIN)/2).T for it in range(ds.n_test_traj_per_point)])
        pstart = np.array([(ds.HMIN+(ds.Y_test_transp[kkk,it].T+1)*(ds.HMAX-ds.HMIN)/2).T for it in range(ds.n_test_traj_per_point)])
        
        
        R = np.array([(ds.HMIN+(ds.X_test_transp[kkk, it].T+1)*(ds.HMAX-ds.HMIN)/2).T for it in range(ds.n_test_traj_per_point)])
        #G = (gen_trajectories[kkk].transpose((0,2,1))*ds.test_std+ds.test_mean).transpose((0,2,1))
        #R = ds.X_test_count[kkk].transpose((0,2,1))
        if kkk==0:
            print(pstart)
            
            print(pstart.shape)
        fig = plt.figure()
        ax = plt.axes(projection ='3d')
        ax.plot3D(G[:,0,:].T, G[:,1,:].T,G[:,2,:].T, color = 'orange', linestyle='solid')
        ax.plot3D(R[:,0,:].T, R[:,1,:].T,R[:,2,:].T, color = 'blue', linestyle='solid')
    

        ax.set_ylabel('y')
        ax.set_xlabel('x')
        #plt.legend()
        plt.tight_layout()
        fig.savefig(plots_path+"/WGAN_"+opt.model_name+"_stoch_rescaled_trajectories_point_"+str(kkk)+".png")
        plt.close()


#PLOT HISTOGRAMS
if HIST_FLAG:
    plt.rcParams.update({'font.size': 25})

    bins = 50
    time_instant = -1
    print("Plotting histograms...")
    for kkk in range(ds.n_points_test):
        fig, ax = plt.subplots(opt.x_dim,1, figsize = (12,opt.x_dim*4))
        for d in range(opt.x_dim):
            G = np.array([np.round(ds.HMIN+(gen_trajectories[kkk, it].T+1)*(ds.HMAX-ds.HMIN)/2).T for it in range(ds.n_test_traj_per_point)])
            R = np.array([np.round(ds.HMIN+(ds.X_test_transp[kkk, it].T+1)*(ds.HMAX-ds.HMIN)/2).T for it in range(ds.n_test_traj_per_point)])
            #G = (gen_trajectories[kkk].transpose((0,2,1))*ds.test_std+ds.test_mean).transpose((0,2,1))
            #R = ds.X_test_count[kkk].transpose((0,2,1))

            XXX = np.vstack((R[:, d,time_instant], G[:, d, time_instant])).T
            
            ax[d].hist(XXX, bins = bins, stacked=False, density=False, color=colors, label=leg)
            ax[d].legend()
            ax[d].set_ylabel(opt.species_labels[d])

        figname = plots_path+"/WGAN_"+opt.model_name+"_rescaled_hist_comparison_{}th_timestep_{}.png".format(time_instant, kkk)
        fig.suptitle('cwgan-gp',fontsize=40)
        fig.savefig(figname)
        plt.tight_layout()
        plt.close()






if WASS_FLAG:
    plt.rcParams.update({'font.size': 22})

    dist = np.zeros(shape=(ds.n_points_test, opt.x_dim, opt.traj_len))
    print("Computing and Plotting Wasserstein distances...") 
    for kkk in tqdm(range(ds.n_points_test)):
        #print("\tinit_state n = ", kkk)
        for m in range(opt.x_dim):
            for t in range(opt.traj_len):    
                A = ds.X_test_transp[kkk,:,m,t]
                B = gen_trajectories[kkk,:,m,t]
                
                dist[kkk, m, t] = wasserstein_distance(A, B)
                

    avg_dist = np.mean(dist, axis=0)
    markers = ['--','-.',':']
    fig = plt.figure()
    for spec in range(opt.x_dim):
        plt.plot(np.arange(opt.traj_len), avg_dist[spec], markers[spec], label=opt.species_labels[spec])
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("wass dist")
    plt.title('cwgan-gp')
    plt.tight_layout()

    figname = plots_path+"/WGAN_"+opt.model_name+"_avg_wass_distance.png"
    fig.savefig(figname)
    plt.close()

    distances_dict = {"gen_hist":B, "ssa_hist":A, "wass_dist":dist}
    file = open(plots_path+f'/WGAN_{opt.model_name}_avg_wass_distances.pickle', 'wb')
    # dump information to that file
    pickle.dump(distances_dict, file)
    # close the file
    file.close()

if not WASS_FLAG:
    plt.rcParams.update({'font.size': 22})

    dist = np.zeros(shape=(ds.n_points_test, opt.x_dim, opt.traj_len))
    print("Computing and Plotting Rescaled Wasserstein distances...") 
    for kkk in range(ds.n_points_test):
        #print("\tinit_state n = ", kkk)
        for t in range(opt.traj_len):    
            
            Gt = np.round(ds.HMIN+(gen_trajectories[kkk, :, :, t]+1)*(ds.HMAX-ds.HMIN)/2)
            Rt = np.round(ds.HMIN+(ds.X_test_transp[kkk, :, :, t]+1)*(ds.HMAX-ds.HMIN)/2)

            for m in range(opt.x_dim):
                A = Rt[m]
                B = Gt[m]
                
                dist[kkk, m, t] = wasserstein_distance(A, B)
                

    avg_dist = np.mean(dist, axis=0)
    markers = ['--','-.',':']
    fig = plt.figure()
    for spec in range(opt.x_dim):
        plt.plot(np.arange(opt.traj_len), avg_dist[spec], markers[spec], label=opt.species_labels[spec])
    plt.legend()
    plt.title('cwgan-gp')
    plt.xlabel("time")
    plt.ylabel("wass dist")
    plt.tight_layout()

    figname = plots_path+"/WGAN_"+opt.model_name+"_rescaled_avg_wass_distance.png"
    fig.savefig(figname)
    distances_dict = {"gen_hist":B, "ssa_hist":A, "wass_dist":dist}
    file = open(plots_path+f'/WGAN_{opt.model_name}_rescaled_avg_wass_distances.pickle', 'wb')
    # dump information to that file
    pickle.dump(distances_dict, file)
    # close the file
    file.close()


