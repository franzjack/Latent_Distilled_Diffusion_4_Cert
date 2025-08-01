import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader

import sys
sys.path.append(".")

class Generator(nn.Module):
    def __init__(self, x_dim, traj_len, latent_dim):
        super(Generator, self).__init__()

        #self.init_size = traj_len // int(traj_len/2)
        self.x_dim = x_dim
        self.padd = 1
        self.n_filters = 2*self.padd+1
        

        self.Q = traj_len

        self.Nch = 256

        self.l1 = nn.Sequential(nn.Linear(latent_dim, self.Nch * self.Q))


        self.conv_blocks = nn.Sequential(

            nn.Conv1d(self.Nch+x_dim*2, 128, kernel_size=3, stride=1, padding=self.padd),  
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
    

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=self.padd),  
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
    
        )

        self.conv_blocks2 = nn.Sequential(
                                          
            nn.Conv1d(256+x_dim*2, 128, kernel_size=3, stride=1, padding=self.padd),  
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=self.padd),  
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
                                             
                                          
        )

        self.conv_blocks3 = nn.Sequential(
                                          
            nn.Conv1d(128+x_dim*2, 128, kernel_size=3, stride=1, padding=self.padd),  
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, x_dim, kernel_size=3, stride=1, padding=self.padd),  
            nn.Tanh(),                                  
                                          
        )
        
    def forward(self, noise, starts, goals):
        c_device = noise.device
        s_cond= starts.to(c_device)
        g_cond= goals.to(c_device)
        #print('q shape', self.Q)
        #print('cond shape:', s_cond.shape)
        s_conds_rep = torch.matmul(s_cond,torch.ones(1, self.Q, device = c_device))
        

        g_conds_rep = torch.matmul(g_cond,torch.ones(1, self.Q, device = c_device))

        noise_out = self.l1(noise).to(c_device)
        noise_out = noise_out.view(noise_out.shape[0], self.Nch, self.Q)
        #print('noise shape:', noise_out.shape)
        b_gen_input = torch.cat((s_conds_rep, noise_out.to(c_device)), 1).to(c_device)
        gen_input = torch.cat((b_gen_input, g_conds_rep), 1).to(c_device)
        #print(gen_input.shape)
        a_traj = self.conv_blocks(gen_input)
        b_a_traj = torch.cat((s_conds_rep, a_traj.to(c_device)), 1).to(c_device)
        gen_input_2 = torch.cat((b_a_traj, g_conds_rep), 1).to(c_device)
        b_traj = self.conv_blocks2(gen_input_2)

        b_b_traj = torch.cat((s_conds_rep, b_traj.to(c_device)), 1).to(c_device)
        gen_input_3 = torch.cat((b_b_traj, g_conds_rep), 1).to(c_device)
        traj = self.conv_blocks3(gen_input_3)

        #print('output shape is:', traj.shape)

        return traj

