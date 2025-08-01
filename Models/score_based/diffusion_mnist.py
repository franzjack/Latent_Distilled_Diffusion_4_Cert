import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Conv2D-based spatiotemporal block
class SpatiotemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 3), padding=(1, 1)):
        super().__init__()
        # Conv2D expects input shape [B, C, H, W]
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(4, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # x shape: [B, C, T]
        B, C, T = x.shape
        x_2d = x.view(B, C, 1, T)  # [B, C, 1, T] (H=1, W=T)
        x_2d = self.conv2d(x_2d)  # [B, out_channels, H, W] = [B, out_channels, 2, T]
        x_2d = self.norm(x_2d)
        x_2d = self.act(x_2d)
        x_flat = x_2d.view(B, -1, x_2d.shape[-1])  # [B, out_channels * H, W=T]
        return x_flat


# class DiffusionEmbedding(nn.Module):
#     def __init__(self, num_steps, dim=128):
#         super().__init__()
#         self.dim = dim
#         self.register_buffer("embedding", self._build_embedding(num_steps, dim), persistent=False)
#         self.proj1 = nn.Linear(dim, dim)
#         self.proj2 = nn.Linear(dim, dim)

#     def _build_embedding(self, num_steps, dim):
#         steps = torch.arange(num_steps).unsqueeze(1).float()
#         freqs = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
#         args = steps * freqs
#         emb = torch.zeros((num_steps, dim))
#         emb[:, 0::2] = torch.sin(args)
#         emb[:, 1::2] = torch.cos(args)
#         return emb

#     def forward(self, diffusion_step):
#         x = self.embedding[diffusion_step]  # (B, dim)
#         x = self.proj1(x)
#         x = F.silu(x)
#         x = self.proj2(x)
#         x = F.silu(x)
#         return x



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
        table = torch.sin(table)  # (T,dim)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_dim, traj_dim):
        super().__init__()
        self.diff_proj = nn.Linear(diffusion_dim, channels)
        self.conv1 = nn.Linear(channels, 2 * channels)
        self.norm1 = nn.GroupNorm(4, 2 * channels)
        self.conv2d_block = SpatiotemporalConvBlock(channels, channels)

    def forward(self, x, diffusion_emb):
        # x: [B, C, T]
        B, C, T = x.shape
        d_emb = self.diff_proj(diffusion_emb).unsqueeze(-1).to(x.device)  # [B, C, 1]
        
        y = x + d_emb  # [B, C, T]

        y = self.conv1(y)  # [B, 2*C, T]
        y = self.norm1(y)
        y = F.silu(y)

        gate = y[:, :C, :]
        filter_ = y[:, C:, :]
        y = torch.sigmoid(gate) * torch.tanh(filter_)

        y = self.conv2d_block(y)  # [B, C * 2, T] since conv2d outputs [B, C_out*H, T]


        y = y[:, :C, :]  # Retain only original number of channels if needed

        return (x + y) / math.sqrt(2), y


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim, traj_len):
        super().__init__()
        self.K = inputdim
        self.L = traj_len
        self.channels = config["channels"]

        self.diff_emb = DiffusionEmbedding(config["num_steps"], config["diffusion_embedding_dim"])

        self.input_proj = nn.Linear(self.K, self.channels)
        self.output_proj1 = nn.Linear(self.channels, self.channels)
        self.output_proj2 = nn.Linear(self.channels, self.K)

        self.res_layers = nn.ModuleList([
            ResidualBlock(self.channels, config["diffusion_embedding_dim"], self.channels)
            for _ in range(config["layers"])
        ])

    def forward(self, x, diffusion_step):
        # x: [B, K, T]
        x = self.input_proj(x)  # [B, channels, T]
        x = F.silu(x)

        d_emb = self.diff_emb(diffusion_step)  # [B, dim]
        skip_sum = 0

        for layer in self.res_layers:
            x, skip = layer(x, d_emb)
            skip_sum = skip_sum + skip

        x = skip_sum / math.sqrt(len(self.res_layers))  # [B, channels, T]
        x = self.output_proj1(x)  # [B, channels, T]
        x = F.silu(x)
        x = self.output_proj2(x)  # [B, K, T]
        return x
