#new main model

import numpy as np
import torch
import torch.nn as nn
import sys
import os

import torch.nn.functional as F
from copy import copy, deepcopy

sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
#from score_based.new_diffusion import diff_CSDI
from score_based.diffusion_mnist import diff_CSDI
from tqdm import tqdm


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device, teacher_model = None, alphas=None):
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
        self.gamma = config_diff["gamma"]
        self.diffmodel = diff_CSDI(config_diff, input_dim, traj_len)
        
        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            beta = torch.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
            alpha_hat = 1 - beta
            alpha = torch.cumprod(alpha_hat,dim=0).to(self.device)
            #self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
            alpha_torch = alpha.float().unsqueeze(1).unsqueeze(1)
            self.register_buffer("alpha", alpha.float())
            self.register_buffer("alpha_torch", alpha.float().unsqueeze(1).unsqueeze(1))
            self.register_buffer("alpha_hat", alpha_hat.float())
            self.register_buffer("beta", beta.float())
        elif config_diff["schedule"] == "linear":
            beta = torch.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            ).to(self.device)

            alpha_hat = 1 - beta
            alpha = torch.cumprod(alpha_hat,dim=0).to(self.device)
            #self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
            alpha_torch = alpha.float().unsqueeze(1).unsqueeze(1)
            self.register_buffer("alpha", alpha.float())
            self.register_buffer("alpha_torch", alpha.float().unsqueeze(1).unsqueeze(1))
            self.register_buffer("alpha_hat", alpha_hat.float())
            self.register_buffer("beta", beta.float())
        
        elif config_diff["schedule"] == "student":
            if teacher_model is None:
                raise ValueError("Teacher model must be provided for student schedule.")
            else:
                alpha = teacher_model.alpha[::2]
                alpha = alpha.to(self.device)
                alpha_hat = torch.empty_like(alpha)
                alpha_hat[0] = alpha[0]
                alpha_hat[1:] = alpha[1:] / alpha[:-1]
                alpha_hat = alpha_hat.to(self.device)

                # beta_student = 1 - alpha_hat_student
                beta = (1 - alpha_hat).to(self.device)
                

                # Volendo, puoi costruire anche la versione torch-friendly come prima
                alpha_torch = alpha.float().unsqueeze(1).unsqueeze(1).to(self.device)
                self.register_buffer("alpha", alpha.float())
                self.register_buffer("alpha_torch", alpha.float().unsqueeze(1).unsqueeze(1))
                self.register_buffer("alpha_hat", alpha_hat.float())
                self.register_buffer("beta", beta.float())
        elif config_diff["schedule"] == "custom":
            if alphas is None:
                raise ValueError("Teacher model must be provided for student schedule.")
            else:
                alpha = alphas
                alpha = alpha.to(self.device)
                alpha_hat = torch.empty_like(alpha)
                alpha_hat[0] = alpha[0]
                alpha_hat[1:] = alpha[1:] / alpha[:-1]
                alpha_hat = alpha_hat.to(self.device)

                # beta_student = 1 - alpha_hat_student
                beta = (1 - alpha_hat).to(self.device)

                alpha_torch = alpha.float().unsqueeze(1).unsqueeze(1).to(self.device)
                self.register_buffer("alpha", alpha.float())
                self.register_buffer("alpha_torch", alpha.float().unsqueeze(1).unsqueeze(1))
                self.register_buffer("alpha_hat", alpha_hat.float())
                self.register_buffer("beta", beta.float())

        



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
    


    def my_loss(self, observed_data, is_train, set_t=-1):
        
        # c = observed_data[:,:,:1]
        # observed_data = observed_data[:,:,1:]
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
        residual = (observed_data - predicted)
        #print('predicted : ', predicted)
        #print('residual sum ',residual.sum())

        x_pred = (noisy_data - (1.0 - current_alpha) ** 0.5 * predicted)/(current_alpha ** 0.5)
        
        #residual_x = (observed_data-x_pred)
        #loss_x = (residual_x **2).sum()
          
        loss = (residual ** 2).sum()  #+ loss_x
        
        return loss





    def my_impute_implicit(self,observed_data, n_samples):
        
        B, K, L = observed_data.shape
        # c = observed_data[:,:,:1]
        
        # observed_data = observed_data[:,:,1:]
        
        

        imputed_samples = torch.zeros(B, n_samples, K, L,device=self.device).to(self.device)

        latent_samples = torch.zeros(B, n_samples, K, L,device=self.device).to(self.device)
        

        for i in range(n_samples):
            # generate noisy observation for unconditional model

            latent_sample = torch.randn_like(observed_data,device=self.device)
            current_sample = copy(latent_sample)
            for t in range(self.num_steps - 1, -1, -1):


                
                #diff_input = torch.cat([c, current_sample], dim=2)
                diff_input = current_sample
                #print('current time is: ', t)
                #print('current value of impute is', current_sample[0])
                
                
                  
                predicted = self.diffmodel(diff_input,t).to(self.device)
                #print('current prediction is: ', predicted[0])
            
                current_sample = self.implicit_single_step(current_sample, predicted, t)
            #imputed_samples[:, i] = torch.cat([c,current_sample.detach()],dim=2)
            imputed_samples[:, i] = current_sample.detach()
            latent_samples[:, i] = latent_sample.detach()
        return imputed_samples, latent_samples
    
    def latent_impute_implicit(self,observed_data, n_samples, latent_samples=None):
        
        B, K, L = observed_data.shape
        #print('observed data shape is: ', observed_data.shape)
        # c = observed_data[:,:,:1]
        
        # observed_data = observed_data[:,:,1:]
        
        

        imputed_samples = torch.zeros(B, n_samples, K, L,device=self.device).to(self.device)
        
        
        

        for i in range(n_samples):
            # generate noisy observation for unconditional model

            latent_sample = latent_samples[:,i]
            latent_sample = latent_sample.squeeze(1)
            current_sample = copy(latent_sample)
            for t in range(self.num_steps - 1, -1, -1):


                
                #diff_input = torch.cat([c, current_sample], dim=2)
                diff_input = current_sample
                #print('current time is: ', t)
                #print('current value of impute is', current_sample[0])
                
                
                  
                predicted = self.diffmodel(diff_input,t).to(self.device)
                #print('current prediction is: ', predicted[0])
            
                current_sample = self.implicit_single_step(current_sample, predicted, t)
            #imputed_samples[:, i] = torch.cat([c,current_sample.detach()],dim=2)
            imputed_samples[:, i] = current_sample.detach()
            latent_samples[:, i] = latent_sample.detach()
        return imputed_samples

    
    def single_impute_implicit(self,current_sample):
            
        current_sample = current_sample.to(self.device)
        for t in range(self.num_steps - 1, -1, -1):
            #diff_input = torch.cat([c, current_sample], dim=2)
            
            #print('current time is: ', t)
            #print('current value of impute is', current_sample[0])
            
            
            predicted = self.diffmodel(current_sample,t).to(self.device)
            #print('current prediction is: ', predicted[0])
        
            current_sample = self.implicit_single_step(current_sample, predicted, t).to(self.device)

        return current_sample

    


    def forward(self, data, is_train=1):

        #ridurre il batch a solo observed data deduce cond_info da dati
        #maybe rendere flessibile il numero di stati iniziali che il modello vede
        


        #elimina side_info


        loss_func = self.my_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(data, is_train)

    def evaluate_implicit(self, batch, n_samples):
        
        observed_data = batch

        with torch.no_grad():


            samples, latent = self.my_impute_implicit(observed_data, n_samples)

            #for i in range(len(cut_length)):  # to avoid double evaluation
            #    target_mask[i, ..., 0 : cut_length[i].item()] = 0
        #print('sample shape is: ', samples.shape)
        #print('observed_shape is: ', observed_data.shape)
        return samples, observed_data, latent
    
    def stud_evaluate_implicit(self, batch, n_samples, latent):
        
        observed_data = batch

        with torch.no_grad():


            samples = self.latent_impute_implicit(observed_data, n_samples,latent)

            #for i in range(len(cut_length)):  # to avoid double evaluation
            #    target_mask[i, ..., 0 : cut_length[i].item()] = 0
        #print('sample shape is: ', samples.shape)
        #print('observed_shape is: ', observed_data.shape)
        return samples, observed_data, latent
    

    def implicit_single_step(self,current_sample, diff_output, t):

        alpha_hat = self.alpha_hat.float().unsqueeze(1).unsqueeze(1).to(self.alpha_torch.device)
        #print(alpha_hat.shape)
        #print(self.alpha_torch.shape)
        
        coeff1 = (self.alpha_torch[t]/alpha_hat[t]) ** 0.5
        coeff2 = 1/(1 - self.alpha_torch[t]) ** 0.5  
        coeff3 = (self.alpha_torch[t])**0.5
        coeff4 = (1 - self.alpha_torch[t]/alpha_hat[t])**0.5
        
        return coeff1*(diff_output) + coeff4*coeff2 * (current_sample - coeff3*diff_output)

    def predicted_clean(self,current_sample, diff_output, t):
        #alpha_hat = self.alpha_hat.float().unsqueeze(1).unsqueeze(1).to(self.alpha_torch.device)
        #print(alpha_hat.shape)
        #print(self.alpha_torch.shape)
        
        coeff1 = (1/self.alpha_torch[t]) ** 0.5
        coeff2 = (1 - self.alpha_torch[t]) ** 0.5  
        #coeff3 = (1 - self.alpha_torch[t]/alpha_hat[t])**0.5
        
        return coeff1*(current_sample - coeff2 * diff_output) 

    def predict_noise(self,current_sample, diff_output, t):
        coeff2 = 1/(1 - self.alpha_torch[t]) ** 0.5  
        coeff3 = (self.alpha_torch[t])**0.5

        return coeff2 * (current_sample - coeff3*diff_output)
        

    def alpha_rec(self, t):
        return(((1- self.alpha_torch[t])/self.alpha_torch[t])**0.5)
    


    def my_distill_loss(self, teacher_model, observed_data):
        gamma = self.gamma
        B, K, L = observed_data.shape
        #print("batch dimension",B)
        t = 2 * torch.randint(1, int((teacher_model.num_steps / 2) ), [B], device=teacher_model.device)
        noise = torch.randn_like(observed_data, device=teacher_model.device)
        #print(teacher_model.alpha_torch[t_teacher])
        t_student = t//2

        t_teacher = t+1
        t_student = t_student.to(self.device)
        t_teacher = t_teacher.to(teacher_model.device)

        #print('alpha student',self.alpha_torch[t_student])

        #print('alpha teacher', teacher_model.alpha_torch[t_teacher])
        noisy_data = (teacher_model.alpha_torch[t_teacher] ** 0.5) * observed_data + ((1.0 - teacher_model.alpha_torch[t_teacher]) ** 0.5) * noise
        self.train() 
        with torch.no_grad():
            # Teacher prediction
            
            x_0_1 = teacher_model.diffmodel(noisy_data, t_teacher) # Teacher denoises at t_teacher

            e_t = teacher_model.predict_noise(noisy_data, x_0_1, t_teacher)
            
            noisy_data1 = (teacher_model.alpha_torch[t_teacher-1] ** 0.5) * x_0_1 + (1.0 - teacher_model.alpha_torch[t_teacher-1]) ** 0.5 * e_t

            x_0_2 = teacher_model.diffmodel(noisy_data1, t_teacher-1)

            e_t1 = teacher_model.predict_noise(noisy_data, x_0_1, t_teacher-1)

            noisy_data2 = (teacher_model.alpha_torch[t_teacher-2] ** 0.5) * x_0_2 + (1.0 - teacher_model.alpha_torch[t_teacher-2]) ** 0.5 * e_t1
            #alpha_hat2 = teacher_model.alpha_torch[t_teacher-1]/teacher_model.alpha_torch[t_teacher-2]
        
        coef = ((1-self.alpha_torch[t_student-1])/(1-self.alpha_torch[t_student]))**0.5
            
        student_target = (noisy_data2 - coef * noisy_data)/((self.alpha_torch[t_student-1])**0.5 - coef*(self.alpha_torch[t_student])**0.5)

        
        x_student = self.diffmodel(noisy_data, t_student)
        #x_student = self.implicit_single_step(noisy_data, e_student, t_student)

        #x_teacher_theo = teacher_model.implicit_single_step(noisy_data, e_student, t_teacher)
        #print(' alpha hat teacher', teacher_model.alpha_hat )
        #print(' alpha hat student', self.alpha_hat )

        #print('rescaling factor', resc)
        #print('prediction of teacher at step 2', e_t1[0])
        #print('denoising of teacher at step 1', x_t_1[0])

        #print('noise target', noise[0])
        #print('t_teacher', t_teacher)
        #print('t student', t_student[0])
        #print('teacher alphas', teacher_model.alpha_torch[t_teacher])
        #print('student alphas' , self.alpha_torch[t_student])
        #print('student_target', student_target[0])
        #print("e_student", e_student[0])
        #print('sum of noise', noise.sum())
        #print('sum of target', student_target.sum())
        #print('theoretical x teacher: ', x_teacher_theo[0])
        #print("noisy data input", noisy_data)
        #print("student x", x_student[0])
        #print("teacher x", x_t_2[0])


        weight = torch.pow((1-self.alpha_torch[t_student])/(self.alpha_torch[t_student]), gamma) if gamma > 0 else 1
        #print('weight is: ', weight)
        # MSE loss between student and teacher's predicted noise
        #loss = F.mse_loss(weight * x_student, weight * x_t1)
        #loss = F.mse_loss(weight * e_student, weight * student_target)

        loss = F.mse_loss(weight * x_student, weight * student_target)

        #print("e_target", student_target)
        #print("e_student", e_student)

        return loss

    def my_distill_loss2(self, teacher_model, observed_data):
        gamma = self.gamma
        B, K, L = observed_data.shape
        #print("batch dimension",B)
        t = 2 * torch.randint(1, int((teacher_model.num_steps / 2) ), [B], device=teacher_model.device)
        noise = torch.randn_like(observed_data, device=teacher_model.device)
        #print(teacher_model.alpha_torch[t_teacher])
        t_student = t//2

        t_teacher = t+1
        t_student = t_student.to(self.device)
        t_teacher = t_teacher.to(teacher_model.device)

        #print('alpha student',self.alpha_torch[t_student])

        #print('alpha teacher', teacher_model.alpha_torch[t_teacher])
        noisy_data = (teacher_model.alpha_torch[t_teacher] ** 0.5) * observed_data + ((1.0 - teacher_model.alpha_torch[t_teacher]) ** 0.5) * noise
        #print('noisy data shape is: ', noisy_data.shape)
        #print('noisy data is: ', noisy_data[0])
        self.train() 
        with torch.no_grad():
            # Teacher prediction
            
            x_0_1 = teacher_model.diffmodel(noisy_data, t_teacher) # Teacher denoises at t_teacher

            noisy_data1 = teacher_model.implicit_single_step(noisy_data, x_0_1, t_teacher)

            x_0_2 = teacher_model.diffmodel(noisy_data1, t_teacher-1)

            noisy_data2 = teacher_model.implicit_single_step(noisy_data1, x_0_2, t_teacher-1)

        coef = ((1-self.alpha_torch[t_student-1])/(1-self.alpha_torch[t_student]))**0.5
            
        student_target = (noisy_data2 - coef * noisy_data)/((self.alpha_torch[t_student-1])**0.5 - coef*(self.alpha_torch[t_student])**0.5)

        
        x_student = self.diffmodel(noisy_data, t_student)
        #x_student = self.implicit_single_step(noisy_data, e_student, t_student)

        #x_teacher_theo = teacher_model.implicit_single_step(noisy_data, e_student, t_teacher)
        #print(' alpha hat teacher', teacher_model.alpha_hat )
        #print(' alpha hat student', self.alpha_hat )

        #print('rescaling factor', resc)
        #print('prediction of teacher at step 2', e_t1[0])
        #print('denoising of teacher at step 1', x_t_1[0])

        #print('noise target', noise[0])
        #print('t_teacher', t_teacher)
        #print('t student', t_student[0])
        #print('teacher alphas', teacher_model.alpha_torch[t_teacher])
        #print('student alphas' , self.alpha_torch[t_student])
        #print('student_target', student_target[0])
        #print("e_student", e_student[0])
        #print('sum of noise', noise.sum())
        #print('sum of target', student_target.sum())
        #print('theoretical x teacher: ', x_teacher_theo[0])
        #print("noisy data input", noisy_data)
        #print("student x", x_student[0])
        #print("teacher x", x_t_2[0])


        weight = torch.pow((1-self.alpha_torch[t_student])/(self.alpha_torch[t_student]), gamma) if gamma > 0 else 1
        #print('weight is: ', weight)
        # MSE loss between student and teacher's predicted noise
        #loss = F.mse_loss(weight * x_student, weight * x_t1)
        #loss = F.mse_loss(weight * e_student, weight * student_target)

        loss = F.mse_loss(weight * x_student, weight * student_target)

        #print("e_target", student_target)
        #print("e_student", e_student)

        return loss
    




    def teacher_impute_implicit(self,observed_data, n_samples, teacher_model):
        B, K, L = observed_data.shape

        imputed_samples_s = torch.zeros(B, n_samples, K, L,device=self.device).to(self.device)

        imputed_samples_t = torch.zeros(B, n_samples, K, L,device=self.device).to(self.device)
        

        for i in range(n_samples):
            # generate noisy observation for unconditional model

            noise = torch.randn_like(observed_data,device=self.device)
            current_sample = torch.randn_like(observed_data)
            current_sample_t = copy(current_sample)

            for t in range(int(self.num_steps) - 1, -1, -1):


                
                diff_input = current_sample
                
                
                  
                predicted = self.diffmodel(diff_input,t).to(self.device)
                #print('current prediction is: ', predicted[0])
            
                current_sample = self.implicit_single_step(current_sample, predicted, t)
            imputed_samples_s[:, i] = copy(current_sample.detach())
            for t in range(int(teacher_model.num_steps) - 1, -1, -1):


                
                diff_input_t = current_sample_t
                
                
                  
                predicted_t = self.diffmodel(diff_input_t,t).to(self.device)
                #print('current prediction is: ', predicted[0])
            
                current_sample_t = self.implicit_single_step(current_sample_t, predicted_t, t)
            
            
            
            imputed_samples_t[:, i] = copy(current_sample_t.detach())
        return imputed_samples_s, imputed_samples_t



    def evaluate_stud_and_teach(self, teacher_model, batch, n_samples):

        observed_data = batch

        with torch.no_grad():


            samples_s, samples_t = self.teacher_impute_implicit(observed_data,n_samples,teacher_model)

        return samples_s, samples_t,  observed_data
    
    def latent_impute(self,observed_data, latent):
        B, K, L = observed_data.shape
        # c = observed_data[:,:,:1]
        
        # observed_data = observed_data[:,:,1:]
        
        

        imputed_samples = torch.zeros(B, len(latent), K, L,device=self.device).to(self.device)
        

        for i in range(len(latent)):
            # generate noisy observation for unconditional model

            current_sample = copy(latent[i])

            for t in range(self.num_steps - 1, -1, -1):


                
                #diff_input = torch.cat([c, current_sample], dim=2)
                diff_input = current_sample
                

                #print(diff_input)

                predicted = self.diffmodel(diff_input,t).to(self.device)
                #print('current prediction is: ', predicted[0])
            
                current_sample = self.implicit_single_step(current_sample, predicted, t)
            #imputed_samples[:, i] = torch.cat([c,current_sample.detach()],dim=2)
            #print(current_sample)
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    
    def evaluate_from_latent(self, batch, latent):

        observed_data = batch

        with torch.no_grad():


            samples = self.latent_impute(observed_data, latent)

        return samples, observed_data

    def latent_builder(self, batch, nsample):
        observed_data = batch

        latent = []

        with torch.no_grad():
            for i in range(nsample):
                latent.append(copy(torch.randn_like(observed_data)))
        
        return(latent)




class absCSDI(CSDI_base):
    def __init__(self, config, device, target_dim, teacher_model=None, alphas=None):
        super(absCSDI, self).__init__(target_dim, config, device, teacher_model=teacher_model, alphas=alphas)

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
        self.device = csdi.device


  def forward(self,noisy_obs):


    return self.csdi.single_impute_implicit(noisy_obs)