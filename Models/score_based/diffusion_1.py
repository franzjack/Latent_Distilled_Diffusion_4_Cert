import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
import os

sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)



# Con based diffusion model

def Conv1D_embedding(in_channels,out_channels, kernel_size):
    padding= int((kernel_size-1)/2)
    layer = nn.Conv1d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv2d_with_init(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    nn.init.kaiming_normal_(layer.weight)
    return layer




def Conv_block(in_channels,out_channels, kernel_size):
    padding= int((kernel_size-1)/2)
    block = nn.Sequential(

            nn.Conv1d(in_channels, 128, kernel_size=3, stride=1, padding=padding),  
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
    

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=padding),  
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
    

            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=padding),  
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
    
            nn.Conv1d(128, out_channels, kernel_size=3, stride=1, padding=padding),  
            
        )
    return block


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim ),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, embedding_dim)
        self.projection2 = nn.Linear(embedding_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=128):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.sin(table)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim, traj_len):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.K = inputdim
        self.L = traj_len


        self.input_projection = Conv_block(self.K, self.K, 3)
        self.output_projection1 = Conv_block(self.K, self.channels, 3)
        self.output_projection2 = Conv_block(self.channels, self.K, 3)
        #nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    K=self.K,
                    L = self.L,
                )
                for _ in range(config["layers"])
            ]
        )



    def forward(self, x, diffusion_step):
        x = self.input_projection(x)
        
        x = F.relu(x)#(B,K,L)

        diffusion_emb = self.diffusion_embedding(diffusion_step) #(1,dim)

        skip_sum = None

        # Iterate through the residual layers
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_emb)
            
            # If skip_sum is None, initialize it with the first skip_connection
            if skip_sum is None:
                skip_sum = skip_connection
            else:
                skip_sum = skip_sum + skip_connection

        # Normalize the skip_sum by the square root of the number of layers
        x = skip_sum / math.sqrt(len(self.residual_layers))
        x = self.output_projection1(x)  # (B,channel,L)
        
        x = F.relu(x)
        
        x = self.output_projection2(x)  # (B,K,L)
        #return x[:,:,1:]
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_embedding_dim,K,L):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, K)
        self.mid_projection = Conv_block(K, 2 * channels, 3)
        self.output_projection = Conv_block(channels, 2 * K, 3)







    def forward(self, x, diffusion_emb):
        #sostituire con conv1D?
        diffusion_emb = diffusion_emb.unsqueeze(-1).to(x.device)  # (B,diffusion_emb_dim,1)

        

        y = x + diffusion_emb #(B, K , L)

        y = self.mid_projection(y)  # (B,2*channel,K*L) (B, 2*channels, L)

        y = F.relu(y)

        # Split y into gate and filter_ parts without using slicing
        gate = y[:,:y.shape[1]//2]
        filter_ = y[:,y.shape[1]//2:]
        
        
        y = torch.sigmoid(gate) * torch.tanh(filter_)

        
        y = self.output_projection(y) #(B, 2*K, L)

        residual = y[:,:y.shape[1]//2]
        skip = y[:,y.shape[1]//2:]

        
        return (x + residual) / math.sqrt(2.0), skip
