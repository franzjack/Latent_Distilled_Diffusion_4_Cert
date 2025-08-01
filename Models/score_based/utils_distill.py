import numpy as np
import torch
from torch.optim import Adam, RAdam
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
#
from model_details import *
import sys
import os

from torch.optim.lr_scheduler import _LRScheduler
import math

from transformers import get_scheduler



class CustomLinearDecayLRScheduler(_LRScheduler):
    def __init__(self, optimizer, final_epoch, last_epoch=-1):
        self.final_epoch = final_epoch

        # Set initial_lr for each param_group if not already set
        for group in optimizer.param_groups:
            if 'initial_lr' not in group:
                group['initial_lr'] = group['lr']

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1

        if epoch == 1:
            lr = 1e-6
        elif epoch == 2:
            lr = 1e-5
        elif epoch == 3:
            lr = 1e-4
        elif epoch == 4:
            lr = 1e-3
        elif epoch == 5:
            lr = 1e-2    
        else:
            # Linear decay from 1e-1 to 1e-5 between epoch 6 and final_epoch
            progress = (epoch - 5) / (self.final_epoch - 5)
            progress = min(max(progress, 0), 1)  # Clamp to [0, 1]
            start = 1e-2
            end = 1e-5
            lr = start + (end - start) * progress

        return [lr for _ in self.optimizer.param_groups]

class CustomSmoothLRScheduler(_LRScheduler):
    def __init__(self, optimizer, final_epoch, last_epoch=-1):
        self.final_epoch = final_epoch

        # Set initial_lr for each param_group if not already set
        for group in optimizer.param_groups:
            if 'initial_lr' not in group:
                group['initial_lr'] = group['lr']

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1  # PyTorch schedulers are zero-based internally

        if epoch == 1:
            lr = 1e-6
        elif epoch == 2:
            lr = 1e-5
        elif epoch == 3:
            lr = 1e-4
        elif epoch == 4:
            lr = 1e-3
        else:
            # Smooth log-linear decay from 1e-1 to 1e-5 between epoch 6 and final_epoch
            progress = (epoch - 5) / (self.final_epoch - 5)
            progress = min(max(progress, 0), 1)  # Clamp between 0 and 1
            log_start = math.log10(1e-3)
            log_end = math.log10(1e-6)
            lr_log = log_start + (log_end - log_start) * progress
            lr = 10 ** lr_log

        return [lr for _ in self.optimizer.param_groups]


def train_student(
    teacher_model,
    student_model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=4,
    foldername="",
):
    #optimizer = RAdam(student_model.parameters(), lr=config["lr"], weight_decay=1e-6)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=config["lr"], weight_decay=1e-6)

    #p1 = int(0.05 * config["epochs"])
    #p2 = int(0.1 * config["epochs"])
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

    losses = []
    val_losses = []

    # p1 = int(0.75 * config["epochs"])
    # p2 = int(0.9 * config["epochs"])
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[p1, p2], gamma=0.1
    # )
    #lr_scheduler = CustomLinearDecayLRScheduler(optimizer, config["epochs"])
    teacher_model.eval()  # Freeze teacher model
    best_valid_loss = 1e10
    student_model.train()
    for epoch_no in range(config["epochs"]):
        avg_loss = 0  # Train student model
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:

            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                # Sample timesteps for distillation
            
                observed_data = student_model.process_data(train_batch)
                # Compute distillation loss
                loss = student_model.my_distill_loss2(teacher_model, observed_data)
                avg_loss += loss.item()
                loss.backward()
                optimizer.step()

                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    refresh=False,
                )

            lr_scheduler.step()

        losses.append(avg_loss / batch_no)

        # Validation
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            student_model.eval()

            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):

                # Sample timesteps for distillation
            
                        observed_data = student_model.process_data(valid_batch)
                        # Compute distillation loss
                        loss = student_model.my_distill_loss2(teacher_model, observed_data)
                        print(loss.item())
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
        if (epoch_no+1)%500 ==0:
            try:
                if foldername != "":
                    temp_path = foldername + str(epoch_no)+"_epoch_model.pth"
                    torch.save(student_model.state_dict(), temp_path)
            except:
                print("failed saving model at epoch:", epoch_no)

    # Save trained student model

    
    if foldername != "":
        output_path = foldername + "student_fullmodel.pth"
        torch.save(student_model.state_dict(), output_path)


    plt.plot(np.arange(len(losses)), losses, color='b', label='train')
    if valid_loader is not None:
        try:
            plt.plot(np.arange(start=0, stop=len(losses), step=valid_epoch_interval), val_losses, color='g', label='valid')
        except:
            print("val losses not available")
    plt.tight_layout()
    plt.title('losses')
    plt.savefig(foldername + 'student_losses.png')
    plt.close()
    

def stud_and_teach_eval(model_s, model_t, test_loader, nsample, scaler=1, mean_scaler=0, foldername="", ds_id = 'test',dist_step = 0):
    print('Evaluation using the implicit version of the model over the '+ds_id+' dataset..')
    with torch.no_grad():
        model_s.eval()
        model_t.eval()
        mse_total_s = 0
        mae_total_s = 0
        mse_total_t = 0
        mae_total_t = 0
        evalpoints_total = 0
        
        all_target = []
        #all_observed_point = []
        #all_observed_time = []
        #all_evalpoint = []
        all_generated_samples_s = []
        all_generated_samples_t = []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            
            for batch_no, test_batch in enumerate(it, start=1):
                #print(test_batch['timepoints'].shape)
                # output = model_s.evaluate_stud_and_teach(model_t,test_batch, nsample)
                # samples_s,samples_t, c_target = output
                latent = model_s.latent_builder(test_batch, nsample)
                samples_s, c_target = model_s.evaluate_from_latent(test_batch, latent)
                samples_t, c_target2 = model_t.evaluate_from_latent(test_batch, latent)
                print(samples_s.shape)
                print(samples_t.shape)
                print(c_target.shape)
                samples_s = samples_s.permute(0, 1, 3, 2)
                samples_t = samples_t.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)


                samples_median = samples_s.median(dim=1)
                all_target.append(c_target)

                all_generated_samples_s.append(samples_s)
                all_generated_samples_t.append(samples_t)

                mse_current_s = (
                    ((samples_s.squeeze(1) - c_target)) ** 2
                ) * (scaler ** 2)

                mse_current_t = (
                    ((samples_t.squeeze(1) - c_target)) ** 2
                ) * (scaler ** 2)

            
                mae_current_s = (
                    torch.abs((samples_s.squeeze(1) - c_target)) 
                ) * scaler

                mae_current_t = (
                    torch.abs((samples_t.squeeze(1) - c_target)) 
                ) * scaler

                #if batch_no%30 == 0:
                #   print('current MSE is: ',mse_current)
                #   print('current MAE is: ',mae_current)
        

                mse_total_s += mse_current_s.sum().item()
                mae_total_s += mae_current_s.sum().item()

                mse_total_t += mse_current_t.sum().item()
                mae_total_t += mae_current_t.sum().item()
                

                it.set_postfix(
                    ordered_dict={
                        "rmse_total student": np.sqrt(mse_total_s),
                        "mae_total student": mae_total_s ,
                        "rmse_total teacher": np.sqrt(mse_total_t),
                        "mae_total teacher": mae_total_t ,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "generated_"+ds_id+f"_step{str(dist_step)}_reshaped_outputs_nsample_distill" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        all_generated_samples_s,
                        all_generated_samples_t,
                        all_target,
                    ],
                    f,
                )
            with open(
                foldername + "generated_"+ds_id+f"_step{str(dist_step)}_outputs_nsample_distill" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)

                all_generated_samples_s = torch.cat(all_generated_samples_s, dim=0)

                all_generated_samples_t = torch.cat(all_generated_samples_t, dim=0)

                pickle.dump(
                    [
                        all_generated_samples_s,
                        all_generated_samples_t,
                        all_target,
                        #all_evalpoint,
                        #all_observed_point,
                        #all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )








def plot_compared_trajs(opt, foldername, dataloader, nsample, idx = 'test', dist_step = 0):
    plt.rcParams.update({'font.size': 25})

    print('Plotting crossroad...')
    path = foldername+'generated_'+idx+f'_step{str(dist_step)}_reshaped_outputs_nsample_distill' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples_s,samples_t, target = pickle.load(f)

    print('student samples len is: ', len(samples_s))
    print('teacher samples len is: ', len(samples_t))
    print('target len is: ', len(target))    

    print('student samples shape is: ', samples_s[0].shape)
    print('teacher samples shape is: ', samples_t[0].shape)
    print('target shape is: ', target[0].shape)
    
    ds = dataloader.dataset
    print('minmax ', ds.min, ds.max)
    colors = ['blue', 'orange', 'green']
    leg = ['real', 'gen_stud', 'gen_teach']

    N = len(samples_s) #nb points

    M = samples_s[0].shape[0] #nb trajs per point
    print('N is:', N)
    print('M is:', M)

        




    G_s = ds.min+(samples_s[0].cpu().numpy()+1)*(ds.max-ds.min)/2 
    G_t = ds.min+(samples_t[0].cpu().numpy()+1)*(ds.max-ds.min)/2 
    R = ds.min+(target[0].cpu().numpy()+1)*(ds.max-ds.min)/2
    
    G_s = G_s.squeeze(1)

    G_t = G_t.squeeze(1)
        
    
    for dataind in range(M):
        fig = plt.figure()
        plt.plot(G_s[dataind,:,0], G_s[dataind,:,1], color = colors[1],linestyle='solid')
        plt.plot(G_t[dataind,:,0], G_t[dataind,:,1], color = colors[2],linestyle='solid')
        plt.plot(R[dataind,:,0], R[dataind,:,1], color = colors[0],linestyle='solid')
    

        plt.ylabel('y')
        plt.xlabel('x')
        #plt.legend()
        plt.tight_layout()
        fig.savefig(foldername+f'CSDI_{opt.model_name}_step{str(dist_step)}_compared_trajs_{idx}_{dataind}.png')
        plt.close()

def plot_compared_many_trajs(opt, foldername, dataloader, nsample, idx = 'test', dist_step = 0):
    plt.rcParams.update({'font.size': 25})

    print('Plotting crossroad...')
    path = foldername+'generated_'+idx+f'_step{str(dist_step)}_reshaped_outputs_nsample_distill' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples_s, samples_t, target = pickle.load(f)

    print('samples len is: ', len(samples_s))
    print('target len is: ', len(target))    

    print('samples shape is: ', samples_s[0].shape)
    print('target shape is: ', target[0].shape)
    
    ds = dataloader.dataset
    print('minmax ', ds.min, ds.max)
    colors = ['blue', 'orange', 'green']
    leg = ['real', 'gen_stud', 'gen_teach']

    N = len(samples_s) #nb points

    M = samples_s[0].shape[0] #nb trajs per point
    print('N is:', N)
    print('M is:', M)

        




    G_s = ds.min+(samples_s[0].cpu().numpy()+1)*(ds.max-ds.min)/2 
    G_t = ds.min+(samples_t[0].cpu().numpy()+1)*(ds.max-ds.min)/2 
    R = ds.min+(target[0].cpu().numpy()+1)*(ds.max-ds.min)/2
    
    G_s = G_s.squeeze(1)
    G_t = G_t.squeeze(1)
        
    fig = plt.figure()
    for dataind in range(M):
        
        plt.plot(G_s[dataind,:,0], G_s[dataind,:,1], color = colors[1],linestyle='solid')
        plt.plot(G_t[dataind,:,0], G_t[dataind,:,1], color = colors[2],linestyle='solid')
        plt.plot(R[dataind,:,0], R[dataind,:,1], color = colors[0],linestyle='solid')
    

    plt.ylabel('y')
    plt.xlabel('x')
    #plt.legend()
    plt.tight_layout()
    fig.savefig(foldername+f'CSDI_{opt.model_name}_step{str(dist_step)}_compared_many_trajs{idx}_.png')
    plt.close()