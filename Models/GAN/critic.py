import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Critic(nn.Module):
    def __init__(self, x_dim, traj_len):
        super(Critic, self).__init__()
        traj_len = traj_len+1


        def critic_block(in_filters, out_filters, L, fil):
            padd = 1
            n_filters = 2*padd + 2 + fil           
            block = [nn.Conv1d(in_filters, out_filters, n_filters, stride=2, padding=padd), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.2)]
            block.append(nn.LayerNorm([out_filters, L]))

            return block
        
        self.model = nn.Sequential(
            *critic_block(x_dim, 64, traj_len//2,2),
            *critic_block(64, 64, traj_len//4,0),
            *critic_block(64,128, traj_len//8,0)
            
        )
        
        

        # The height and width of downsampled image
        
        ds_size = (traj_len +1) // (2**2)

        self.adv_layer = nn.Sequential(nn.Linear(64 * ds_size, 1), 
                                       )
        
    def forward(self, trajs, conditions, goals):
        d_in = torch.cat((conditions, trajs), 2)
        d_in = torch.cat((d_in,goals),2)
        #print('d_in shape is ', d_in.shape)
        out = self.model(d_in)
        #print('shape before flat is :',out.shape)
        out_flat = out.view(out.shape[0], -1)
        #print('shape before prob is :',out_flat.shape)
        validity = self.adv_layer(out_flat)
        return validity