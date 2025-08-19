import numpy as np
import torch
from torch.optim import Adam, RAdam
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from torchvision.utils import make_grid
import pandas as pd
#
from model_details import *
import sys
import os


from transformers import get_scheduler


sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
#import torch_two_sample 

def train(
    model,
    config,
    train_loader,
    autoencoder=None,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
):

    if foldername != "":
        output_path1 = foldername + "fullmodel.pth"
        output_path2 = foldername + "diffmodel.pth"

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)

    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2, eta_min=0.00001)

    #p1 = int(0.1 * config["epochs"])
    #p2 = int(0.2 * config["epochs"])
    #p3 = int(0.3 * config["epochs"])
    #p4 = int(0.4 * config["epochs"])
    p5 = int(0.5 * config["epochs"])
    #p6 = int(0.6 * config["epochs"])
    p7 = int(0.7 * config["epochs"])
    #p8 = int(0.8 * config["epochs"])
    p9 = int(0.9 * config["epochs"])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p5, p7, p9], gamma=0.1
    )


    # optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # num_training_steps = len(train_loader) * config["epochs"]
    # num_warmup_steps = int(0.05 * num_training_steps)  # 5% warmup

    # lr_scheduler = get_scheduler(
    #     name="linear",  # or "linear"
    #     optimizer=optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_training_steps
    # )


    losses = []
    val_losses = []

    

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train() # setting the model to training mode
        for x, _ in train_loader:
            x: torch.Tensor = x.to(model.device)

            autoencoder.encoder.eval()  # setting the autoencoder to evaluation mode
            xx = autoencoder.encoder(x)  # encoding the input

            xx = xx.unsqueeze(-1)  # adding a new dimension for the diffusion model
            


            optimizer.zero_grad()
            
            loss = model(xx)
            #print('current loss is ',loss)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            print("avg epoch loss: ", avg_loss)
        lr_scheduler.step()

        losses.append(avg_loss)
        #valid_loader = None
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval() # setting the model to evaluation mode
            avg_loss_valid = 0
            with torch.no_grad(): 
                for x, _ in valid_loader:
                    x: torch.Tensor = x.to(model.device)

                    autoencoder.encoder.eval()  # setting the autoencoder to evaluation mode
                    xx = autoencoder.encoder(x)  # encoding the input

                    xx = xx.unsqueeze(-1)

                    loss = model(xx, is_train=0)  # calculating the loss in evaluation mode
                    avg_loss_valid += loss.item()

                    print("valid_epoch_loss: ", loss.item())

            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid,
                    "at",
                    epoch_no,
                )
            val_losses.append(avg_loss_valid)

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
        torch.save(model.state_dict(), output_path1)
        #check se sensato
        torch.save(model.diffmodel.state_dict(), output_path2)



def train_mnist(
    model,
    config,
    train_loader,
    autoencoder=None,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
):

    if foldername != "":
        output_path1 = foldername + "fullmodel.pth"
        output_path2 = foldername + "diffmodel.pth"

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)

    # Multi-step learning rate decay
    p5 = int(0.5 * config["epochs"])
    p7 = int(0.7 * config["epochs"])
    p9 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p5, p7, p9], gamma=0.1
    )

    losses = []
    val_losses = []
    best_valid_loss = float("inf")

    for epoch_no in range(config["epochs"]):
        avg_loss = 0.0
        model.train()
        autoencoder.encoder.eval()  # Move outside batch loop

        with tqdm(train_loader, desc=f"Epoch {epoch_no+1}/{config['epochs']}",
                  leave=True, dynamic_ncols=True, mininterval=0.5) as it:

            for batch_no, (x, _) in enumerate(it, start=1):
                x = x.to(model.device)

                # Encode input
                xx = autoencoder.encoder(x)
                xx = xx.unsqueeze(-1)

                # Forward + backward
                optimizer.zero_grad()
                loss = model(xx)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                it.set_postfix(loss=loss.item(), avg_loss=avg_loss / batch_no)

        lr_scheduler.step()
        losses.append(avg_loss)

        # Validation phase
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            autoencoder.encoder.eval()
            avg_loss_valid = 0.0

            with torch.no_grad(), tqdm(valid_loader, desc=f"Validation {epoch_no+1}",
                                       leave=True, dynamic_ncols=True, mininterval=0.5) as val_it:
                for val_batch_no, (x, _) in enumerate(val_it, start=1):
                    x = x.to(model.device)

                    xx = autoencoder.encoder(x).unsqueeze(-1)
                    loss = model(xx, is_train=0)
                    avg_loss_valid += loss.item()

                    val_it.set_postfix(val_loss=loss.item(),
                                       avg_val_loss=avg_loss_valid / val_batch_no)

            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(f"\n✨ Best validation loss updated to {avg_loss_valid:.4f} at epoch {epoch_no+1}")

            val_losses.append(avg_loss_valid)

    # Plotting training and validation losses
    plt.plot(np.arange(len(losses)), losses, color='b', label='train')
    if valid_loader is not None:
        plt.plot(np.arange(start=0, stop=len(losses), step=valid_epoch_interval), val_losses, color='g', label='valid')
    plt.tight_layout()
    plt.title('Losses')
    plt.legend()
    plt.savefig(foldername + 'losses.png')
    plt.close()

    # Save models
    if foldername != "":
        torch.save(model.state_dict(), output_path1)
        torch.save(model.diffmodel.state_dict(), output_path2)


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


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", ds_id = 'test'):
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
                

def new_evaluate_implicit(model, test_loader, autoencoder=None, nsample=100, scaler=1, mean_scaler=0, foldername="", ds_id = 'test'):
    print('Evaluation using the implicit version of the model over the '+ds_id+' dataset..')
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        #all_observed_point = []
        #all_observed_time = []
        #all_evalpoint = []
        all_generated_samples = []
        for x, _ in test_loader:
            x: torch.Tensor = x.to(model.device)
                #print(test_batch['timepoints'].shape)

            xx = autoencoder.encoder(x)  # encoding the input

            xx = xx.unsqueeze(-1)
            output = model.evaluate_implicit(xx, nsample)
            samples, c_target, latent = output
            print('samples shape is: ', samples.shape)
            print('c_target shape is: ', c_target.shape)
            print('latent shape is: ', latent.shape)
            samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
            c_target = c_target.permute(0, 2, 1)  # (B,L,K)
            #eval_points = eval_points.permute(0, 2, 1)
            #observed_points = observed_points.permute(0, 2, 1)

            samples_median = samples.median(dim=1)
            all_target.append(c_target)
            #all_evalpoint.append(eval_points)
            #all_observed_point.append(observed_points)
            #all_observed_time.append(observed_time)
            all_generated_samples.append(samples)

            mse_current = (
                ((samples.squeeze(1) - c_target)) ** 2
            ) * (scaler ** 2)

        
            mae_current = (
                torch.abs((samples.squeeze(1) - c_target)) 
            ) * scaler

            #if batch_no%30 == 0:
                #   print('current MSE is: ',mse_current)
                #   print('current MAE is: ',mae_current)
    

            mse_total += mse_current.sum().item()
            mae_total += mae_current.sum().item()
            
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
            foldername + "sampled_"+ds_id+"_latent_nsample" + str(nsample) + ".pk", "wb"
        ) as f:
            pickle.dump(
                [
                    latent,
                ],
                f,
            )
        with open(
            foldername + "generated_"+ds_id+"_outputs_nsample" + str(nsample) + ".pk", "wb"
        ) as f:
            all_target = torch.cat(all_target, dim=0)
            #all_evalpoint = torch.cat(all_evalpoint, dim=0)
            #all_observed_point = torch.cat(all_observed_point, dim=0)
            #all_observed_time = torch.cat(all_observed_time, dim=0)
            all_generated_samples = torch.cat(all_generated_samples, dim=0)

            pickle.dump(
                [
                    all_generated_samples,
                    all_target,
                    #all_evalpoint,
                    #all_observed_point,
                    #all_observed_time,
                    scaler,
                    mean_scaler,
                ],
                f,
            )

def new_evaluate_implicit_mnist(model, test_loader, autoencoder=None, nsample=100, scaler=1, mean_scaler=0, foldername="", ds_id='test'):
    print(f"Evaluation using the implicit version of the model over the {ds_id} dataset...")

    with torch.no_grad():
        model.eval()
        autoencoder.encoder.eval()

        mse_total = 0.0
        mae_total = 0.0

        all_target = []
        all_generated_samples = []

        with tqdm(test_loader, desc=f"Evaluating {ds_id}", dynamic_ncols=True, mininterval=0.5, leave=True) as it:
            for batch_no, (x, _) in enumerate(it, start=1):
                x = x.to(model.device)

                xx = autoencoder.encoder(x)
                xx = xx.unsqueeze(-1)

                output = model.evaluate_implicit(xx, nsample)
                samples, c_target, latent = output

                # Optional debug prints
                # print('samples shape:', samples.shape)
                # print('c_target shape:', c_target.shape)
                # print('latent shape:', latent.shape)

                samples = samples.permute(0, 1, 3, 2)  # (B, nsample, L, K)
                c_target = c_target.permute(0, 2, 1)    # (B, L, K)

                samples_median = samples.median(dim=1)  # (B, L, K)
                all_target.append(c_target)
                all_generated_samples.append(samples)

                mse_current = ((samples.squeeze(1) - c_target) ** 2) * (scaler ** 2)
                mae_current = torch.abs(samples.squeeze(1) - c_target) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()

                it.set_postfix(mse=mse_current.mean().item(), mae=mae_current.mean().item())

        # Save reshaped outputs
        with open(f"{foldername}generated_{ds_id}_reshaped_outputs_nsample{nsample}.pk", "wb") as f:
            pickle.dump([all_generated_samples, all_target], f)

        # Save latent samples
        with open(f"{foldername}sampled_{ds_id}_latent_nsample{nsample}.pk", "wb") as f:
            pickle.dump([latent], f)

        # Save final outputs (concatenated)
        with open(f"{foldername}generated_{ds_id}_outputs_nsample{nsample}.pk", "wb") as f:
            all_target = torch.cat(all_target, dim=0)
            all_generated_samples = torch.cat(all_generated_samples, dim=0)

            pickle.dump(
                [
                    all_generated_samples,
                    all_target,
                    scaler,
                    mean_scaler,
                ],
                f,
            )


def stud_evaluate_implicit(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", ds_id = 'test',latent=None):
    print('Evaluation using the implicit version of the model over the '+ds_id+' dataset..')
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        #all_observed_point = []
        #all_observed_time = []
        #all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            
            for batch_no, test_batch in enumerate(it, start=1):
                #print(test_batch['timepoints'].shape)
                output = model.stud_evaluate_implicit(test_batch, nsample,latent)
                samples, c_target, latent = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                #eval_points = eval_points.permute(0, 2, 1)
                #observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                #all_evalpoint.append(eval_points)
                #all_observed_point.append(observed_points)
                #all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples.squeeze(1) - c_target)) ** 2
                ) * (scaler ** 2)

            
                mae_current = (
                    torch.abs((samples.squeeze(1) - c_target)) 
                ) * scaler

                #if batch_no%30 == 0:
                 #   print('current MSE is: ',mse_current)
                 #   print('current MAE is: ',mae_current)
        

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total ),
                        "mae_total": mae_total ,
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
                foldername + "sampled_"+ds_id+"_latent_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        latent,
                    ],
                    f,
                )
            with open(
                foldername + "generated_"+ds_id+"_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                #all_evalpoint = torch.cat(all_evalpoint, dim=0)
                #all_observed_point = torch.cat(all_observed_point, dim=0)
                #all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        #all_evalpoint,
                        #all_observed_point,
                        #all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            


                                
                
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
    
    wass_dist = np.empty((N, K, L))
    
    for i in range(N):
        
        Ti = np.round(ds.min+(target[i].cpu().numpy()+1)*(ds.max-ds.min)/2)
        Si = np.round(ds.min+(samples[i].cpu().numpy()+1)*(ds.max-ds.min)/2)        
            
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



def plot_results(opt, foldername, autoencoder, nsample, idx = 'test'):
    plt.rcParams.update({'font.size': 25})

    print('Plotting images...')
    path = foldername+'generated_'+idx+'_reshaped_outputs_nsample' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples, target = pickle.load(f)

    print('samples len is: ', len(samples))
    print('target len is: ', len(target))    

    print('samples shape is: ', samples[0].shape)
    print('target shape is: ', target[0].shape)
    

    N = len(samples) #nb points

    M = samples[0].shape[0] #nb trajs per point
    print('N is:', N)
    print('M is:', M)

    samples = samples[0].squeeze(1)

    samples = samples.squeeze(-1)
    target = target[0].squeeze(-1)



    
    
        
    
    for dataind in range(M):
        fig = plt.figure()
        x = autoencoder.decoder(samples[dataind])
        x2 = autoencoder.decoder(target[dataind])



        grid1 = make_grid(x).permute(1, 2, 0).cpu().detach().numpy()
        grid2 = make_grid(x2).permute(1, 2, 0).cpu().detach().numpy()

        # Concatenate vertically (axis=0) or horizontally (axis=1)
        combined = np.concatenate((grid1, grid2), axis=0)  # vertical comparison
        # combined = np.concatenate((grid1, grid2), axis=1)  # horizontal comparison

        # Plot and save
        plt.figure(figsize=(x.shape[0], 4))
        plt.imshow(combined)
        plt.axis('off')
        plt.tight_layout()
        save_path = foldername + f'CSDI_mnist_comparison_{dataind}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plot_rescaled_many_trajs(opt, foldername, dataloader, nsample, idx = 'test'):
    plt.rcParams.update({'font.size': 25})

    print('Plotting crossroad...')
    path = foldername+'generated_'+idx+'_reshaped_outputs_nsample' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples, target = pickle.load(f)

    print('samples len is: ', len(samples))
    print('target len is: ', len(target))    

    print('samples shape is: ', samples[0].shape)
    print('target shape is: ', target[0].shape)
    
    ds = dataloader.dataset
    print('minmax ', ds.min, ds.max)
    colors = ['blue', 'orange']
    leg = ['real', 'gen']

    N = len(samples) #nb points

    M = samples[0].shape[0] #nb trajs per point
    print('N is:', N)
    print('M is:', M)

        




    G = ds.min+(samples[0].cpu().numpy()+1)*(ds.max-ds.min)/2 
    R = ds.min+(target[0].cpu().numpy()+1)*(ds.max-ds.min)/2
    
    G = G.squeeze(1)
        
    fig = plt.figure()
    for dataind in range(M):
        
        plt.plot(G[dataind,:,0], G[dataind,:,1], color = colors[1],linestyle='solid')
        plt.plot(R[dataind,:,0], R[dataind,:,1], color = colors[0],linestyle='solid')
    

    plt.ylabel('y')
    plt.xlabel('x')
    #plt.legend()
    plt.tight_layout()
    fig.savefig(foldername+f'CSDI_{opt.model_name}_crossroad_many_trajs{idx}_.png')
    plt.close()


def plot_rescaled_trajectories(opt, foldername, dataloader, nsample, Mred = 10):
    plt.rcParams.update({'font.size': 25})
    
    if opt.implicit_flag==True:
        sched_type='implicit'
    else:
        sched_type='markovian'

    print('Plotting rescaled trajectories...')
    path = foldername+'generated_test_reshaped_outputs_nsample' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples, target = pickle.load(f)

    ds = dataloader.dataset

    colors = ['blue', 'orange']
    leg = ['real', 'gen']

    N = len(samples) #nb points
    M = samples[0].shape[0] #nb trajs per point
    K = samples[0].shape[-1] #feature
    L = samples[0].shape[-2] #time length
    print(L)
    tspan = range(int(-opt.testmissingratio),L)
    print(tspan)
    print(K)
    print(Mred)
    for dataind in range(N):

        G = np.round(ds.min+(samples[dataind][:,0].cpu().numpy()+1)*(ds.max-ds.min)/2)     
        R = np.round(ds.min+(target[dataind].cpu().numpy()+1)*(ds.max-ds.min)/2)        
        fig, axes = plt.subplots(K,figsize=(16, K*4))
        
        G[:,:int(-opt.testmissingratio)] = R[:,:int(-opt.testmissingratio)].copy()
        
        for kk in range(K):
            if K == 1:
                for jj in range(Mred):
                    if jj == 0:
                        axes.plot(tspan, G[jj,int(-opt.testmissingratio):,kk], color = colors[1],linestyle='solid',label=leg[1])
                        axes.plot(tspan, R[jj,int(-opt.testmissingratio):,kk], color = colors[0],linestyle='solid',label=leg[0])
                        print('done!')
                    else:
                        axes.plot(tspan, G[jj,int(-opt.testmissingratio):,kk], color = colors[1],linestyle='solid')
                        axes.plot(tspan, R[jj,int(-opt.testmissingratio):,kk], color = colors[0],linestyle='solid')
                        
                axes.set_ylabel(opt.species_labels[kk])
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
        fig.savefig(foldername+f'CSDI_{opt.model_name}_stoch_'+sched_type+'_rescaled_trajectories_point_{}.png'.format(dataind))
        plt.close()



def plot_rescaled_3dline(opt, foldername, dataloader, nsample, idx = 'test'):
    plt.rcParams.update({'font.size': 25})

    print('Plotting 3d line...')
    path = foldername+'generated_'+idx+'_reshaped_outputs_nsample' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples, target = pickle.load(f)

    print('samples len is: ', len(samples))
    print('target len is: ', len(target))    

    print('samples shape is: ', samples[0].shape)
    print('target shape is: ', target[0].shape)
    
    ds = dataloader.dataset
    print('minmax ', ds.min, ds.max)
    colors = ['blue', 'orange']
    leg = ['real', 'gen']

    N = len(samples) #nb points

    M = samples[0].shape[0] #nb trajs per point
    print('N is:', N)
    print('M is:', M)

        




    G = ds.min+(samples[0].cpu().numpy()+1)*(ds.max - ds.min)/2 
    R = ds.min+(target[0].cpu().numpy()+1)*(ds.max - ds.min)/2
    
    G = G.squeeze(1)
        
    
    for dataind in range(M):

        fig = plt.figure()
        ax = plt.axes(projection ='3d')
        ax.plot3D(G[dataind,:,0], G[dataind,:,1], G[dataind,:,2], color = colors[1], linestyle='solid')
        ax.plot3D(R[dataind,:,0], R[dataind,:,1], R[dataind,:,2], color = colors[0], linestyle='solid')
    

        ax.set_ylabel('y')
        ax.set_xlabel('x')
        #plt.legend()
        plt.tight_layout()
        fig.savefig(foldername+f'CSDI_{opt.model_name}_3dmap_{idx}_{dataind}.png')
        plt.close()




def train_student_mnist(
    teacher_model,
    student_model,
    config,
    train_loader,
    autoencoder=None,
    valid_loader=None,
    valid_epoch_interval=4,
    foldername="",
):
    optimizer = torch.optim.Adam(student_model.parameters(), lr=config["lr"], weight_decay=1e-6)

    p5 = int(0.5 * config["epochs"])
    p7 = int(0.7 * config["epochs"])
    p9 = int(0.9 * config["epochs"])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p5, p7, p9], gamma=0.1
    )

    losses = []
    val_losses = []

    teacher_model.eval()
    best_valid_loss = 1e10
    student_model.train()
    autoencoder.encoder.eval()
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        

        with tqdm(train_loader, desc=f"Epoch {epoch_no+1}", dynamic_ncols=True, mininterval=0.5) as it:
            for batch_no, (x, _) in enumerate(it, start=1):
                x = x.to(student_model.device)

                # === VAE ENCODING ===
                z = autoencoder.encoder(x)
                z = z.unsqueeze(-1)  # Add temporal dimension for diffusion

                optimizer.zero_grad()
                observed_data = z
                loss = student_model.my_distill_loss2(teacher_model, observed_data)

                avg_loss += loss.item()
                loss.backward()
                optimizer.step()

                it.set_postfix({
                    "avg_epoch_loss": avg_loss / batch_no,
                    "lr": optimizer.param_groups[0]["lr"]
                }, refresh=True)

        lr_scheduler.step()
        losses.append(avg_loss / batch_no)

        # Validation
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            student_model.eval()
            avg_loss_valid = 0

            with torch.no_grad(), tqdm(valid_loader, desc=f"Validation {epoch_no+1}", dynamic_ncols=True, mininterval=0.5) as it:
                for batch_no, (x, _) in enumerate(it, start=1):
                    x = x.to(student_model.device)

                    # === VAE ENCODING ===
                    autoencoder.encoder.eval()
                    z = autoencoder.encoder(x)
                    z = z.unsqueeze(-1)

                    observed_data = z
                    loss = student_model.my_distill_loss2(teacher_model, observed_data)
                    avg_loss_valid += loss.item()

                    it.set_postfix({
                        "valid_avg_epoch_loss": avg_loss_valid / batch_no
                    }, refresh=True)

            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(f"\n✨ Best valid loss updated to {avg_loss_valid / batch_no:.4f} at epoch {epoch_no+1}")

            val_losses.append(avg_loss_valid / batch_no)

        # Optional: save periodically
        if (epoch_no + 1) % 500 == 0 and foldername:
            temp_path = foldername + f"{epoch_no}_epoch_model.pth"
            try:
                torch.save(student_model.state_dict(), temp_path)
            except Exception as e:
                print(f"Failed saving model at epoch {epoch_no}: {e}")

    if foldername:
        output_path = foldername + "student_fullmodel.pth"
        torch.save(student_model.state_dict(), output_path)

    # Plotting losses
    plt.plot(np.arange(len(losses)), losses, color='b', label='train')
    if valid_loader is not None and val_losses:
        try:
            plt.plot(np.arange(start=0, stop=len(losses), step=valid_epoch_interval), val_losses, color='g', label='valid')
        except Exception as e:
            print(f"Error plotting validation losses: {e}")
    plt.tight_layout()
    plt.title('Losses')
    plt.legend()
    plt.savefig(foldername + 'student_losses.png')
    plt.close()




def stud_and_teach_eval_mnist(model_s, model_t, test_loader, nsample, scaler=1, autoencoder=None, mean_scaler=0, foldername="", ds_id='test', dist_step=0):
    print(f'Evaluation using student and teacher over the {ds_id} dataset...')
    
    with torch.no_grad():
        model_s.eval()
        model_t.eval()
        
        mse_total_s = 0
        mae_total_s = 0
        mse_total_t = 0
        mae_total_t = 0

        all_target = []
        all_generated_samples_s = []
        all_generated_samples_t = []

        with tqdm(test_loader, desc="Evaluating", dynamic_ncols=True, mininterval=0.5) as it:
            for batch_no, (x, _) in enumerate(it, start=1):
                x = x.to(model_s.device)
                xx = autoencoder.encoder(x)
                xx = xx.unsqueeze(-1)
                print('xx shape is: ', xx.shape)
                #latent = torch.randn_like(xx)  # Assuming latent is a random tensor for evaluation
                latent = model_s.latent_builder(xx, nsample)
                #print('latent shape is', latent.shape)
                samples_s, c_target = model_s.evaluate_from_latent(xx, latent)
                samples_t, _ = model_t.evaluate_from_latent(xx, latent)

                samples_s = samples_s.permute(0, 1, 3, 2)
                samples_t = samples_t.permute(0, 1, 3, 2)
                c_target = c_target.permute(0, 2, 1)

                all_target.append(c_target)
                all_generated_samples_s.append(samples_s)
                all_generated_samples_t.append(samples_t)

                mse_current_s = ((samples_s.squeeze(1) - c_target) ** 2) * (scaler ** 2)
                mae_current_s = torch.abs(samples_s.squeeze(1) - c_target) * scaler

                mse_current_t = ((samples_t.squeeze(1) - c_target) ** 2) * (scaler ** 2)
                mae_current_t = torch.abs(samples_t.squeeze(1) - c_target) * scaler

                mse_total_s += mse_current_s.sum().item()
                mae_total_s += mae_current_s.sum().item()

                mse_total_t += mse_current_t.sum().item()
                mae_total_t += mae_current_t.sum().item()

                it.set_postfix({
                    "rmse_s": np.sqrt(mse_total_s),
                    "mae_s": mae_total_s,
                    "rmse_t": np.sqrt(mse_total_t),
                    "mae_t": mae_total_t
                }, refresh=True)

        # Save reshaped samples
        with open(f"{foldername}generated_{ds_id}_step{dist_step}_reshaped_outputs_nsample_distill{nsample}.pk", "wb") as f:
            pickle.dump([all_generated_samples_s, all_generated_samples_t, all_target], f)

        # Save final samples (flattened)
        with open(f"{foldername}generated_{ds_id}_step{dist_step}_outputs_nsample_distill{nsample}.pk", "wb") as f:
            pickle.dump([
                torch.cat(all_generated_samples_s, dim=0),
                torch.cat(all_generated_samples_t, dim=0),
                torch.cat(all_target, dim=0),
                scaler,
                mean_scaler,
            ], f)


def plot_compared_results(opt, foldername, autoencoder, nsample, idx='test', dist_step=0):
    plt.rcParams.update({'font.size': 25})
    print('Plotting student vs teacher reconstructions...')

    path = os.path.join(foldername, f'generated_{idx}_step{dist_step}_reshaped_outputs_nsample_distill{nsample}.pk')
    with open(path, 'rb') as f:
        samples_s, samples_t, _ = pickle.load(f)

    print(f'student samples len: {len(samples_s)}')
    print(f'teacher samples len: {len(samples_t)}')

    print('student sample shape:', samples_s[0].shape)
    print('teacher sample shape:', samples_t[0].shape)

    samples_s = samples_s[0].squeeze(1).squeeze(-1)  # (M, latent_dim)
    samples_t = samples_t[0].squeeze(1).squeeze(-1)  # (M, latent_dim)

    M = samples_s.shape[0]
    print('Number of comparisons:', M)

    for dataind in range(M):
        decoded_stud = autoencoder.decoder(samples_s[dataind])
        decoded_teach = autoencoder.decoder(samples_t[dataind])

        # Make image grids
        grid_s = make_grid(decoded_stud).permute(1, 2, 0).cpu().detach().numpy()
        grid_t = make_grid(decoded_teach).permute(1, 2, 0).cpu().detach().numpy()

        # Combine vertically (student on top, teacher below)
        combined = np.concatenate((grid_s, grid_t), axis=0)

        plt.figure(figsize=(grid_s.shape[1] / 10, 5))
        plt.imshow(combined)
        plt.axis('off')
        plt.tight_layout()
        save_path = os.path.join(foldername, f'CSDI_{opt.model_name}_step{dist_step}_student_vs_teacher_{idx}_{dataind}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
