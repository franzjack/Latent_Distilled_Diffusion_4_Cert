import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Conv1D-based spatiotemporal block
class SpatiotemporalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden_channels = 32  # You can adjust this based on your needs
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.act1 = nn.SiLU()
        self.norm1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2)
        self.act2 = nn.SiLU()
        self.norm2 = nn.BatchNorm1d(hidden_channels * 2)

        self.conv3 = nn.Conv1d(hidden_channels * 2, out_channels, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm1d(out_channels)

    def forward(self, x):  # x: [B, C, T]
        x = self.conv1(x)
        x = self.act1(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        return x


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim),
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

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.sin(table)  # (T,dim)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_dim, traj_dim):
        super().__init__()
        self.diff_proj = nn.Linear(diffusion_dim, channels)
        self.conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(2 * channels)
        self.conv2d_block = SpatiotemporalConv1D(channels, channels)

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

        y = self.conv2d_block(y)  # [B, C, T]

        return (x + y) / math.sqrt(2), y


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim, traj_len):
        super().__init__()
        self.K = inputdim
        self.L = traj_len
        self.channels = config["channels"]

        self.diff_emb = DiffusionEmbedding(config["num_steps"], config["diffusion_embedding_dim"])

        self.input_proj = nn.Conv1d(self.K, self.channels, kernel_size=3, padding=1)
        self.output_proj1 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.output_proj2 = nn.Conv1d(self.channels, self.K, kernel_size=3, padding=1)

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
