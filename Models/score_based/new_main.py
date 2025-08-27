#new main model

import numpy as np
import torch
import torch.nn as nn
import sys
import os

sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from score_based.diffusion_2 import diff_CSDI

from tqdm import tqdm


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
        
        #self.Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor


        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = config_diff["input_dim"]
        traj_len = config_diff["traj_len"]
        self.diffmodel = diff_CSDI(config_diff, input_dim, traj_len)
        
        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = torch.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = torch.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            ).to(self.device)

        self.alpha_hat = 1 - self.beta
        self.alpha = torch.cumprod(self.alpha_hat,dim=0).to(self.device)
        #self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        self.alpha_torch = self.alpha.float().unsqueeze(1).unsqueeze(1)



    def calc_loss_valid(
        self, observed_data, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.my_loss(
                observed_data, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps
    


    def my_loss(self, observed_data, is_train, set_t=-1, is_cond=False):
        
        #c = observed_data[:,:,:1]
        #observed_data = observed_data[:,:,1:]
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B,device=self.device) * set_t).long()
        else:
            t = torch.randint(0, self.num_steps, [B], device = self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data, device = self.device)
        #print('noise shape is ', noise.shape)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        #total_input = torch.cat([c, noisy_data], dim=2)
        total_input = noisy_data
        #print('the current shape of step is ', t.shape)
        predicted = self.diffmodel(total_input,t) #(B,K,L)
        residual = (noise - predicted)
        #print('predicted : ', predicted)
        #print('residual sum ',residual.sum())
        
        num_eval = len(noise)  
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        
        return loss



    def calc_loss(
        self, observed_data, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B,device=self.device) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B],device=self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data,device=self.device)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        #qui sostituisci con cat perché saranno solo due cose


        mask = torch.zeros_like(observed_data,device=self.device)
        mask[:,:1] = 1
        cond_obs = (mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1) 

        B, inputdim, K, L = total_input.shape
        total_input = total_input.reshape(B, inputdim, K * L)
        
        

        

        predicted = self.diffmodel(total_input,t)  # (B,K,L)

        predicted = predicted.reshape(B, K, L)


        #qui sarà verosimilmente solo noise e predicted
        residual = (noise - predicted)
        #num_eval per noi sarà costante
        num_eval = len(noise) -1
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss



    def my_impute_implicit(self,observed_data, n_samples):
        c = observed_data[:,:,:1]
        observed_data = observed_data[:,:,1:]
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L+1,device=self.device).to(self.device)
        

        for i in range(n_samples):
            # generate noisy observation for unconditional model

            current_sample = torch.randn_like(observed_data,device=self.device)

            for t in range(self.num_steps - 1, -1, -1):


                
                diff_input = torch.cat([c, current_sample], dim=2)
                
                
                  
                predicted = self.diffmodel(diff_input,t).to(self.device)

            
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha[t]) ** 0.5
                
                

                if t > 0:
                    noise = torch.randn_like(current_sample,device = self.device)
                    
                    sigma2 = 0
                    #((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) 
                    
                    coeff3 = (1 - self.alpha[t - 1] - sigma2) ** 0.5
                    current_sample = coeff1 * (current_sample - coeff2 * predicted) + predicted*coeff3
                    current_sample += noise * sigma2 ** 0.5 
                else:
                    current_sample = coeff1 * (current_sample - coeff2 * predicted) + predicted*(1 - self.alpha[t]/self.alpha_hat[t])**0.5
            imputed_samples[:, i] = torch.cat([c,current_sample.detach()],dim=2)
        return imputed_samples
    



    def impute_implicit(self, observed_data, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L,device=self.device)
        cond_mask = torch.zeros_like(observed_data,device=self.device)
        cond_mask[:,:1] = 1

        for i in range(n_samples):
            # generate noisy observation for unconditional model

            current_sample = torch.randn_like(observed_data,device=self.device)

            for t in range(self.num_steps - 1, -1, -1):


                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)
                
                B, inputdim, K, L = diff_input.shape
                diff_input = diff_input.reshape(B, inputdim, K * L)
                
                
                  
                predicted = self.diffmodel(diff_input,t).to(self.device)

                predicted = predicted.reshape(B, K, L)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha[t]) ** 0.5
                
                

                if t > 0:
                    noise = torch.randn_like(current_sample,device=self.device)
                    
                    sigma2 = 0
                    #((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) 
                    
                    coeff3 = (1 - self.alpha[t - 1] - sigma2) ** 0.5
                    current_sample = coeff1 * (current_sample - coeff2 * predicted) + predicted*coeff3
                    current_sample += noise * sigma2 ** 0.5 
                else:
                    current_sample = coeff1 * (current_sample - coeff2 * predicted) + predicted*(1 - self.alpha[t]/self.alpha_hat[t])**0.5
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples
    

    def singleS_impute_implicit(self,observed_data, noisy_obs):
        c = observed_data[:,:,:1].to(self.device)
        observed_data = observed_data[:,:,1:].to(self.device)
        

        #imputed_samples = torch.zeros(B, 1, K, L+1).to(self.device)

        current_sample = noisy_obs.to(self.device)

        for t in range(self.num_steps - 1, -1, -1):


            
            diff_input = torch.cat([c, current_sample], dim=2)
                
                
                  
            predicted = self.diffmodel(diff_input,t).to(self.device)
            coeff1 = 1 / self.alpha_hat[t] ** 0.5
            coeff2 = (1 - self.alpha[t]) ** 0.5                

            if t > 0:
                #sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) 
                
                coeff3 = (1 - self.alpha[t - 1]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted) + predicted*coeff3
            else:
                current_sample = coeff1 * (current_sample - coeff2 * predicted) + predicted*(1 - self.alpha[t]/self.alpha_hat[t])**0.5
            imputed_samples = torch.cat([c,current_sample],dim=2)
        return imputed_samples
        
    def single_impute_implicit(self,observed_data, current_sample):
        c = observed_data[:,:,:1].to(self.device)
        observed_data = observed_data[:,:,1:].to(self.device)
        

       

        for t in range(self.num_steps - 1, -1, -1):
            c = c.to(self.device)
            current_sample = current_sample.to(self.device)



            diff_input = torch.cat([c, current_sample], dim=2).to(self.device)
                
                
                  
            predicted = self.diffmodel(diff_input,t).to(self.device)
            coeff1 = 1 / self.alpha_hat[t] ** 0.5
            coeff2 = (1 - self.alpha[t]) ** 0.5                

            if t > 0:
                #sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) 
                
                coeff3 = (1 - self.alpha[t - 1]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted) + predicted*coeff3
            else:
                current_sample = coeff1 * (current_sample - coeff2 * predicted) + predicted*(1 - self.alpha[t]/self.alpha_hat[t])**0.5
            imputed_samples = torch.cat([c,current_sample],dim=2)
        return imputed_samples

    def single_impute_implicit_unc(self,observed_data, current_sample):
        #c = observed_data[:,:,:1].to(self.device)
        #observed_data = observed_data[:,:,1:].to(self.device)
        for t in range(self.num_steps - 1, -1, -1):
            #c = c.to(self.device)
            current_sample = current_sample.to(self.device)
            #diff_input = torch.cat([c, current_sample], dim=2).to(self.device)
            diff_input = current_sample.to(self.device)   
            predicted = self.diffmodel(diff_input,t).to(self.device)
            coeff1 = 1 / self.alpha_hat[t] ** 0.5
            coeff2 = (1 - self.alpha[t]) ** 0.5                
            if t > 0:
                #sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t])  
                coeff3 = (1 - self.alpha[t - 1]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted) + predicted*coeff3
            else:
                current_sample = coeff1 * (current_sample - coeff2 * predicted) + predicted*(1 - self.alpha[t]/self.alpha_hat[t])**0.5
            #imputed_samples = torch.cat([c,current_sample],dim=2)
            imputed_samples = current_sample
        return imputed_samples

    def forward(self, batch, is_train=1):

        #ridurre il batch a solo observed data deduce cond_info da dati
        #maybe rendere flessibile il numero di stati iniziali che il modello vede

        #observed_data = self.process_data(batch)
        observed_data = batch


        #elimina side_info


        loss_func = self.my_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, is_train)

    def evaluate_implicit(self, batch, n_samples):
        
        #observed_data = self.process_data(batch)
        observed_data = batch
        with torch.no_grad():


            samples = self.my_impute_implicit(observed_data, n_samples)

            #for i in range(len(cut_length)):  # to avoid double evaluation
            #    target_mask[i, ..., 0 : cut_length[i].item()] = 0
        #print('sample shape is: ', samples.shape)
        #print('observed_shape is: ', observed_data.shape)
        return samples, observed_data
    





class absCSDI(CSDI_base):
    def __init__(self, config, device, target_dim=3):
        super(absCSDI, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        #qui avremo solo observed traj e condition
        observed_data = batch["observed_data"].to(self.device).float()
        #print('observed_data traj shape', observed_data.shape)
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        #print('observed_data shape is: ', observed_data.shape)

        #transposing data
        #observed_data = observed_data.permute(0, 2, 1)
        #observed_mask = observed_mask.permute(0, 2, 1)
        #gt_mask = gt_mask.permute(0, 2, 1)

        #cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        #for_pattern_mask = observed_mask

        return (
            observed_data.permute(0,2,1))
    


class Generator(nn.Module):

  def __init__(self, csdi):
        super(Generator, self).__init__()
        self.csdi = csdi


  def forward(self,noisy_obs):


    return self.csdi.single_impute_implicit_unc(noisy_obs,noisy_obs)