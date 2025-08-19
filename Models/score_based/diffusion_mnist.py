import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Minimal 1D Conv Block (pointwise only)
class Conv1DPointwise(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):  # [B, C, T]
        return self.act(self.norm(self.conv(x)))


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

    def forward(self, diffusion_step):  # diffusion_step: [B]
        x = self.embedding[diffusion_step]  # [B, dim]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=128):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        return torch.sin(table)  # (T,dim)


class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_dim):
        super().__init__()
        self.diff_proj = nn.Linear(diffusion_dim, channels)
        self.conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(2 * channels)
        self.conv2 = Conv1DPointwise(channels, channels)

    def forward(self, x, diffusion_emb):
        # x: [B, C, T]
        B, C, T = x.shape
        d_emb = self.diff_proj(diffusion_emb).unsqueeze(-1).to(x.device)  # [B, C, 1]

        y = x + d_emb
        y = self.conv1(y)  # [B, 2*C, T]
        y = self.norm1(y)
        y = F.silu(y)

        # Gating mechanism
        gate, filter_ = y[:, :C, :], y[:, C:, :]
        y = torch.sigmoid(gate) * torch.tanh(filter_)

        y = self.conv2(y)  # [B, C, T]
        return (x + y) / math.sqrt(2), y


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2, traj_len=1):
        super().__init__()
        self.K = inputdim  # latent dim = 2
        self.L = traj_len  # treat as "sequence length"
        self.channels = config["channels"]

        self.diff_emb = DiffusionEmbedding(
            config["num_steps"], config["diffusion_embedding_dim"]
        )

        # Projection in/out (1x1 conv acts like Linear)
        self.input_proj = nn.Conv1d(self.K, self.channels, kernel_size=1)
        self.output_proj1 = nn.Conv1d(self.channels, self.channels, kernel_size=1)
        self.output_proj2 = nn.Conv1d(self.channels, self.K, kernel_size=1)

        self.res_layers = nn.ModuleList([
            ResidualBlock(self.channels, config["diffusion_embedding_dim"])
            for _ in range(config["layers"])
        ])

    def forward(self, x, diffusion_step):
        # x: [B, K=2, T]
        x = self.input_proj(x)  # [B, channels, T]
        x = F.silu(x)

        d_emb = self.diff_emb(diffusion_step)  # [B, dim]
        skip_sum = 0

        for layer in self.res_layers:
            x, skip = layer(x, d_emb)
            skip_sum = skip_sum + skip

        x = skip_sum / math.sqrt(len(self.res_layers))  # [B, channels, T]
        x = self.output_proj1(x)
        x = F.silu(x)
        x = self.output_proj2(x)  # back to [B, 2, T]
        return x
