import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import pandas as pd
#from model_details import *
import sys
import os


sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import torch_two_sample 

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "model.pth"

    losses = []
    val_losses = []

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train() # setting the model to training mode
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:

            for batch_no, train_batch in enumerate(it, start=1):

                
                optimizer.zero_grad()
                
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()

        losses.append(avg_loss / batch_no)
        valid_loader = None
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval() # setting the model to evaluation mode
            avg_loss_valid = 0
            with torch.no_grad(): 
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
            val_losses.append(avg_loss_valid / batch_no)
    
    # plotting the loss function
    fig = plt.plot()
    plt.plot(np.arange(len(losses)), losses, color='b', label='train')
    if valid_loader is not None:
        plt.plot(np.arange(start=0,stop=len(losses),step=valid_epoch_interval), val_losses, color='g', label='valid')
    plt.tight_layout()
    plt.title('losses')
    plt.savefig(foldername+'losses.png')
    plt.close()
    
    if foldername != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def evaluate(model, test_loader, nsample=1, scaler=1, mean_scaler=0, foldername="", ds_id = 'test'):
    print('Evaluation over the '+ds_id+' dataset..')
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            
            for batch_no, test_batch in enumerate(it, start=1):
                #print(test_batch['timepoints'].shape)
                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "generated_"+ds_id+"_reshaped_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                    ],
                    f,
                )
            with open(
                foldername + "generated_"+ds_id+"_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + ds_id+"_result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)

def light_evaluate(model, test_loader, nsample=1, foldername="", ds_id = 'test'):
    with torch.no_grad():
        model.eval()
        n_gen_trajs = nsample
        real_samples = []
        gen_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=150.0) as it:
            
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output
                #print('ZZZZ = ', samples.shape, c_target.shape)
                samples = samples.permute(0, 1, 3, 2)
                #samples = samples[:,0]  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)

                real_samples.append(c_target)
                gen_samples.append(samples)

        N = len(gen_samples) # nb of points
        M = gen_samples[0].shape[0]  #nsamples
        K = gen_samples[0].shape[-1] #feature
        L = gen_samples[0].shape[-2] #time length


        gen_samples_res = torch.zeros((N,M*n_gen_trajs,L,K))

        for i in range(N):
            for j in range(n_gen_trajs):

                gen_samples_res[i,j*M:(j+1)*M] = gen_samples[i][:,j]
        
        with open(
                foldername + "generated_"+ds_id+"_reshaped_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        gen_samples_res, real_samples
                    ],
                    f,
                )
    return gen_samples_res, real_samples#_res

def generate_trajectories(model, test_loader, nsample=1):
    with torch.no_grad():
        model.eval()
        n_gen_trajs = nsample
        gen_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=150.0) as it:
            
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)
                
                gen_samples.append(samples)


        N = len(gen_samples) # nb of points
        M = gen_samples[0].shape[0]  #nsamples
        K = gen_samples[0].shape[-1] #feature
        L = gen_samples[0].shape[-2] #time length


        gen_samples_res = torch.zeros((N,M*n_gen_trajs,L,K))

        for i in range(N):
            for j in range(n_gen_trajs):

                gen_samples_res[i,j*M:(j+1)*M] = gen_samples[i][:,j]
        
        return gen_samples_res



def statistical_test(opt, model, dataloader, nsample=1, scaler=1, mean_scaler=0, foldername=""):
    plt.rcParams.update({'font.size': 22})

    print("Statistical test...") 
    
    path = foldername+'generated_test_reshaped_outputs_nsample' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples, target = pickle.load(f)


    N = len(samples) #nb points
    M = samples[0].shape[0] #nb trajs per point
    K = samples[0].shape[-1] #feature
    L = samples[0].shape[-2] #time length
    
    samples_grid = [5,10, 25, 50, 100, 200, 300]
    G = len(samples_grid)

    file = open(plots_path+f'/CSDI_{opt.model_name}_pvalues.pickle', 'wb')
        
    if True:
        pvals_mean = np.empty((K,G))
        pvals_std = np.empty((K,G))
        for jj in range(G):

            pvals = torch.empty((N, K, L))
            
            for i in range(N):
                Ti = target[i]#.cpu().numpy()
                Si = samples[i]#.cpu().numpy()
                for m in range(K):
                    for t in range(L):    
                        A = Ti[:samples_grid[jj],t,m].unsqueeze(1)
                        B = Si[:samples_grid[jj],0,t,m].unsqueeze(1)
                        st = torch_two_sample.statistics_diff.EnergyStatistic(samples_grid[jj], samples_grid[jj])
                        stat, dist = st(A,B, ret_matrix = True)
                        pvals[i,m,t] = st.pval(dist)

            pvals_mean[:,jj] = np.mean(np.mean(pvals.cpu().numpy(), axis=0),axis=1)
            pvals_std[:,jj] = np.std(np.std(pvals.cpu().numpy(), axis=0),axis=1)
    
        pvalues_dict = {"samples_grid":samples_grid, "pvals_mean":pvals_mean, "pvals_std":pvals_std}
    
        pickle.dump(pvalues_dict, file)
        file.close()
    else:
        with open(file, 'rb') as f:
            pvalues_dict = pickle.load(f)
        pvals_mean, pvals_std = pvalues_dict["pvals_mean"], pvalues_dict["pvals_std"]

    colors = ['b','r','g']
    fig = plt.figure()
    for spec in range(K):
        plt.plot(np.array(samples_grid), pvals_mean[spec], color =colors[spec],label=opt.species_labels[spec])
        plt.fill_between(np.array(samples_grid), pvals_mean[spec]-1.96*pvals_std[spec],
                                    pvals_mean[spec]+1.96*pvals_std[spec], color =colors[spec],alpha =0.1)
    plt.plot(np.array(samples_grid), np.ones(G)*0.05,'k--') 
    plt.legend()
    plt.title(f'csdi: {opt.model_name}')
    plt.xlabel("nb of samples")
    plt.ylabel("p-values")
    plt.tight_layout()
    figname = foldername+f"CSDI_{opt.model_name}_statistical_test.png"
    fig.savefig(figname)
    plt.close()


def compute_wass_distance(opt, model, dataloader, nsample=1, scaler=1, mean_scaler=0, foldername=""):
    plt.rcParams.update({'font.size': 22})

    print("Computing and Plotting Wasserstein distances...") 
    
    path = foldername+'generated_test_reshaped_outputs_nsample' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples, target = pickle.load(f)


    N = len(samples) #nb points
    M = samples[0].shape[0] #nb trajs per point
    K = samples[0].shape[-1] #feature
    L = samples[0].shape[-2] #time length
    
    wass_dist = np.empty((N, K, L))
    
    for i in range(N):
        Ti = target[i].cpu().numpy()
        Si = samples[i].cpu().numpy()
        for m in range(K):
            for t in range(L):    
                A = Ti[:,t,m]
                B = Si[:,0,t,m]
                wd = wasserstein_distance(A, B)
                wass_dist[i,m,t] = wd
                
    avg_dist = np.mean(wass_dist, axis=0)
    markers = ['--','-.',':']
    fig = plt.figure()
    for spec in range(K):
        plt.plot(np.arange(int(-opt.testmissingratio),L), avg_dist[spec][int(-opt.testmissingratio):], markers[spec],label=opt.species_labels[spec])
    plt.legend()
    plt.title('csdi')
    plt.xlabel("time")
    plt.ylabel("wass dist")
    plt.tight_layout()
    figname = foldername+f"CSDI_{opt.model_name}_avg_wass_distance.png"
    fig.savefig(figname)
    plt.close()
    
    return wass_dist

def compute_rescaled_wass_distance(opt, model, dataloader, nsample=1, scaler=1, mean_scaler=0, foldername=""):
    plt.rcParams.update({'font.size': 22})

    print("Computing and Plotting Rescaled Wasserstein distances...") 
    
    path = foldername+'generated_test_reshaped_outputs_nsample' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples, target = pickle.load(f)

    ds = dataloader.dataset

    N = len(samples) #nb points
    M = samples[0].shape[0] #nb trajs per point
    K = samples[0].shape[-1] #feature
    L = samples[0].shape[-2] #time length
    
    wass_dist = np.empty((N, K-1, L))
    
    for i in range(N):
        
        Ti = np.round(ds.min+(target[i].cpu().numpy()+1)*(ds.max-ds.min)/2)
        Si = np.round(ds.min+(samples[i].cpu().numpy()+1)*(ds.max-ds.min)/2)        
            
        for m in range(K-1):
            for t in range(L):    
                A = Ti[:,t,m]
                B = Si[:,0,t,m]
                wd = wasserstein_distance(A, B)
                wass_dist[i,m,t] = wd
                
    avg_dist = np.mean(wass_dist, axis=0)
    markers = ['--','-.',':']
    fig = plt.figure()
    for spec in range(K-1):
        plt.plot(np.arange(int(-opt.testmissingratio),L), avg_dist[spec][int(-opt.testmissingratio):], markers[spec],label=opt.species_labels[spec])
    plt.legend()
    plt.title('csdi')
    plt.xlabel("time")
    plt.ylabel("wass dist")
    plt.tight_layout()
    figname = foldername+f"CSDI_{opt.model_name}_rescaled_avg_wass_distance.png"
    fig.savefig(figname)
    plt.close()
    
    return wass_dist

def plot_histograms(opt, foldername, dataloader, nsample, idx='test'):
    plt.rcParams.update({'font.size': 25})


    ds = dataloader.dataset

    print('Plotting histograms...')
    path = foldername+'generated_'+idx+'_reshaped_outputs_nsample' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples, target = pickle.load(f)

    colors = ['blue', 'orange']
    leg = ['real', 'gen']

    N = len(samples) #nb points
    if idx == 'cal':
        N = 20
    M = samples[0].shape[0] #nb trajs per point
    K = samples[0].shape[-1] #feature
    L = samples[0].shape[-2] #time length

    
    bins = 50
    time_instant = -1 #last time step

    if K == 1:
        
        for kkk in range(N):
            fig, ax = plt.subplots(K,1, figsize = (12,K*4))
            #G = np.round(ds.min+(samples[kkk][:,0].cpu().numpy()+1)*(ds.max-ds.min)/2)     
            #R = np.round(ds.min+(target[kkk].cpu().numpy()+1)*(ds.max-ds.min)/2)        
            G = ds.min+(samples[kkk][:,0].cpu().numpy()+1)*(ds.max-ds.min)/2   
            R = ds.min+(target[kkk].cpu().numpy()+1)*(ds.max-ds.min)/2       
            
            
            for d in range(K):

                XXX = np.vstack((R[:, time_instant, d], G[:, time_instant, d])).T
                
                ax.hist(XXX, bins = bins, stacked=False, density=False, color=colors, label = leg)
                ax.legend()
                ax.set_ylabel(opt.species_labels[d])

            figname = foldername+"CSDI_{}_{}_rescaled_hist_comparison_{}th_timestep_{}.png".format(opt.model_name,idx,time_instant, kkk)
            fig.suptitle('csdi',fontsize=30)
            plt.tight_layout()
            fig.savefig(figname)
            plt.close()
        
    else:
        
        for kkk in range(N):
            fig, ax = plt.subplots(K,1, figsize = (12,K*4))
            #G = np.round(ds.min+(samples[kkk][:,0].cpu().numpy()+1)*(ds.max-ds.min)/2)     
            #R = np.round(ds.min+(target[kkk].cpu().numpy()+1)*(ds.max-ds.min)/2)    
            G = ds.min+(samples[kkk][:,0].cpu().numpy()+1)*(ds.max-ds.min)/2   
            R = ds.min+(target[kkk].cpu().numpy()+1)*(ds.max-ds.min)/2  
            
            for d in range(K):

                XXX = np.vstack((R[:, time_instant, d], G[:, time_instant, d])).T
                
                ax[d].hist(XXX, bins = bins, stacked=False, density=False, color=colors)
                #ax[d].legend()
                ax[d].set_ylabel(opt.species_labels[d])

            figname = foldername+"CSDI_{}_{}_rescaled_hist_comparison_{}th_timestep_{}.png".format(opt.model_name,idx,time_instant, kkk)
            fig.suptitle('csdi',fontsize=40)
            fig.savefig(figname)
            plt.close()
        

def get_quantile(samples,q,dim=1):
    return torch.quantile(samples,q,dim=dim).cpu().numpy()

def plot_rescaled_crossroads(opt, foldername, dataloader, nsample, idx = 'test'):
    plt.rcParams.update({'font.size': 25})

    print('Plotting crossroad...')
    path = foldername+'generated_'+idx+'_reshaped_outputs_nsample' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples, target = pickle.load(f)
    
    ds = dataloader.dataset
    print('minmax ', ds.min, ds.max)
    colors = ['blue', 'orange']
    leg = ['real', 'gen']

    N = len(samples) #nb points

    M = samples[0].shape[0] #nb trajs per point

    
    fig = plt.figure()
    for i in range(N):
        G = ds.min+(samples[i].cpu().numpy()+1)*(ds.max-ds.min)/2 
        R = ds.min+(target[i].cpu().numpy()+1)*(ds.max-ds.min)/2
        
        G[:,:int(-opt.testmissingratio)] = R[:,:int(-opt.testmissingratio)].copy()
            
        
        for dataind in range(M):
            
            plt.plot(G[dataind,:,0], G[dataind,:,1], color = colors[1],linestyle='solid')
            plt.plot(R[dataind,:,0], R[dataind,:,1], color = colors[0],linestyle='solid')

    plt.ylabel('y')
    plt.xlabel('x')
    #plt.legend()
    plt.tight_layout()
    fig.savefig(foldername+f'CSDI_{opt.model_name}_crossroad_{idx}.png')
    plt.close()


def plot_rescaled_trajectories(opt, foldername, dataloader, nsample, idx = 'test', Mred = 1):
    plt.rcParams.update({'font.size': 25})

    print('Plotting rescaled trajectories...')
    path = foldername+'generated_'+idx+'_reshaped_outputs_nsample' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples, target = pickle.load(f)

    ds = dataloader.dataset

    colors = ['blue', 'orange']
    leg = ['real', 'gen']

    N = len(samples) #nb points
    if idx == 'cal':
        N = 20
    M = samples[0].shape[0] #nb trajs per point
    K = samples[0].shape[-1] #feature
    L = samples[0].shape[-2] #time length
    tspan = range(int(-opt.testmissingratio),L)
    for dataind in range(N):
        #G = np.round(ds.min+(samples[dataind][:,0].cpu().numpy()+1)*(ds.max-ds.min)/2)     
        #R = np.round(ds.min+(target[dataind].cpu().numpy()+1)*(ds.max-ds.min)/2)        

        G = ds.min+(samples[dataind].cpu().numpy()+1)*(ds.max-ds.min)/2 
        R = ds.min+(target[dataind].cpu().numpy()+1)*(ds.max-ds.min)/2

        fig, axes = plt.subplots(K,figsize=(16, K*4))
        
        G[:,:int(-opt.testmissingratio)] = R[:,:int(-opt.testmissingratio)].copy()
        
        for kk in range(K):
            if K == 1:
                for jj in range(Mred):
                    if jj == 0:
                        axes.plot(tspan, G[jj,int(-opt.testmissingratio):,kk], color = colors[1],linestyle='solid',label=leg[1])
                        axes.plot(tspan, R[jj,int(-opt.testmissingratio):,kk], color = colors[0],linestyle='solid',label=leg[0])
                        
                    else:
                        axes.plot(tspan, G[jj,int(-opt.testmissingratio):,kk], color = colors[1],linestyle='solid')
                        axes.plot(tspan, R[jj,int(-opt.testmissingratio):,kk], color = colors[0],linestyle='solid')
                        
                axes.set_ylabel('value')
                axes.set_xlabel('time')

            else:

                for jj in range(Mred):
                    if jj == 0:
                        axes[kk].plot(tspan, G[jj,int(-opt.testmissingratio):,kk], color = colors[1],linestyle='solid',label=leg[1])
                        axes[kk].plot(tspan, R[jj,int(-opt.testmissingratio):,kk], color = colors[0],linestyle='solid',label=leg[0])
                        
                    else:
                        axes[kk].plot(tspan, G[jj,int(-opt.testmissingratio):,kk], color = colors[1],linestyle='solid')
                        axes[kk].plot(tspan, R[jj,int(-opt.testmissingratio):,kk], color = colors[0],linestyle='solid')
                        
                plt.setp(axes[kk], ylabel=opt.species_labels[kk])
                if kk == (K-1):
                    plt.setp(axes[kk], xlabel='time')
        plt.legend()
        fig.suptitle('csdi',fontsize=40)
        plt.tight_layout()
        fig.savefig(foldername+f'CSDI_{opt.model_name}_stoch_{idx}_rescaled_trajectories_point_{dataind}.png')
        plt.close()

def avg_stl_satisfaction(opt, foldername, dataloader, model_name, ds_id = 'test', nsample=1):
    plt.rcParams.update({'font.size': 22})

    ds = dataloader.dataset
    print('Computing STL satisfaction over the '+ds_id+ ' set...')

    colors = ['blue', 'orange']
    leg = ['real', 'gen']

    path = foldername+'generated_'+ds_id+'_reshaped_outputs_nsample' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples, target = pickle.load(f)

    N = len(samples) #nb points
    M = samples[0].shape[0] #nb trajs per point
    K = samples[0].shape[-1] #feature
    L = samples[0].shape[-2] #time length

    ssa_sat = np.empty(N)
    gen_sat = np.empty(N)
    for i in range(N):
        #print("\tinit_state n = ", i)
        
        rescaled_samples = np.round(ds.min+(samples[i].cpu().numpy()+1)*(ds.max-ds.min)/2)     
        rescaled_target = np.round(ds.min+(target[i].cpu().numpy()+1)*(ds.max-ds.min)/2)        
        rescaled_samples[:,0,:int(-opt.testmissingratio)] = rescaled_target[:,:int(-opt.testmissingratio)].copy()
        
        ssa_trajs_i = torch.tensor(rescaled_target.transpose((0,2,1)))[:,:,int(-opt.testmissingratio-1):]
        gen_trajs_i = torch.tensor(rescaled_samples[:,0].transpose((0,2,1)))[:,:,int(-opt.testmissingratio-1):]

        if model_name == 'SIR':
            ssa_sat_i = eval_sir_property(ssa_trajs_i,rob_flag).float()
            gen_sat_i = eval_sir_property(gen_trajs_i,rob_flag).float()
        elif model_name == 'eSIRS':
            ssa_sat_i = eval_esirs_property(ssa_trajs_i,rob_flag).float()
            gen_sat_i = eval_esirs_property(gen_trajs_i,rob_flag).float()
        elif model_name == 'TS':
            ssa_sat_i = eval_ts_property(ssa_trajs_i,rob_flag).float()
            gen_sat_i = eval_ts_property(gen_trajs_i,rob_flag).float()
        elif model_name == 'Toy':
            ssa_sat_i = eval_toy_property(ssa_trajs_i,rob_flag).float()
            gen_sat_i = eval_toy_property(gen_trajs_i,rob_flag).float()
        elif model_name == 'Oscillator':
            sum_value = torch.sum(ssa_trajs_i,dim=1)
            ssa_sat_i = eval_oscillator_property(ssa_trajs_i,sum_value,rob_flag).float()
            gen_sat_i = eval_oscillator_property(gen_trajs_i,sum_value,rob_flag).float()
        elif model_name == 'MAPK':
            ssa_sat_i = eval_mapk_property(ssa_trajs_i,rob_flag).float()
            gen_sat_i = eval_mapk_property(gen_trajs_i,rob_flag).float()
        elif model_name == 'EColi':
            sum_value = torch.sum(ssa_trajs_i,dim=1)
            ssa_sat_i = eval_ecoli_property(ssa_trajs_i,sum_value,rob_flag).float()
            gen_sat_i = eval_ecoli_property(gen_trajs_i,sum_value,rob_flag).float()
        elif model_name == "FixedPID":
            ssa_sat_i = eval_aircraft_property(ssa_trajs_i,rob_flag).float()
            gen_sat_i = eval_aircraft_property(gen_trajs_i,rob_flag).float()
        else:
            ssa_sat_i, gen_sat_i = [0],[0]
        ssa_sat[i] = ssa_sat_i.mean().detach().cpu().numpy()
        gen_sat[i] = gen_sat_i.mean().detach().cpu().numpy()



    fig = plt.figure()
    plt.plot(np.arange(N), ssa_sat, 'o-', color=colors[0], label=leg[0])
    plt.plot(np.arange(N), gen_sat, 'o-', color=colors[1], label=leg[1])
    plt.legend()
    plt.title("csdi")
    plt.xlabel("test points")
    if rob_flag:
        plt.ylabel("exp. robustness")
    else:
        plt.ylabel("exp. satisfaction")

    plt.tight_layout()
    if rob_flag:
        figname_stl = foldername+"csdi_"+ds_id+"_stl_quantitative_satisfaction.png"
    else:
        figname_stl = foldername+"csdi_"+ds_id+"_stl_boolean_satisfaction.png"

    fig.savefig(figname_stl)
    plt.close()

    init_states = np.array([target[i].cpu().numpy()[0,0] for i in range(N)])
    sat_diff = np.absolute(ssa_sat-gen_sat)#/(ssa_sat+1e-16)
    
    dist_dict = {'init': init_states, 'sat_diff':sat_diff}
    if rob_flag:
        file = open(foldername+f'quantitative_satisf_distances_'+ds_id+'_set_active={opt.active_flag}.pickle', 'wb')
    else:
        file = open(foldername+f'boolean_satisf_distances_'+ds_id+'_set_active={opt.active_flag}.pickle', 'wb')
    pickle.dump(dist_dict, file)
    file.close()

    return dist_dict


def generate(model, loader, nsample=100, scaler=1, mean_scaler=0, foldername="", extra_str = ""):

    with torch.no_grad():
        model.eval()
        
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(loader, mininterval=5.0, maxinterval=50.0) as it:
            
            for batch_no, test_batch in enumerate(it, start=1):

                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output
                
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                evalpoints_total += eval_points.sum().item()


            with open(
                foldername + "/generated_"+extra_str+"_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_posint = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)
                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )
