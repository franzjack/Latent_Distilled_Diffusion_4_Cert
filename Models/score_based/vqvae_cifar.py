import torch
import torch.nn as nn
import torch.nn.functional as F

class VQVAE(nn.Module):
    def __init__(self, latent_dim=8, codebook_size=512, commitment_cost=0.25):
        super().__init__()
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, 1)
        )
        
        # Codebook
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        self.codebook.weight.data.uniform_(-1/codebook_size, 1/codebook_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        
        # Quantize
        z_flattened = z.permute(0, 2, 3, 1).reshape(-1, self.latent_dim)
        distances = (torch.sum(z_flattened**2, dim=1, keepdim=True) + 
                    torch.sum(self.codebook.weight**2, dim=1) -
                    2 * torch.matmul(z_flattened, self.codebook.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.codebook(encoding_indices).view(z.shape)
        
        # Losses
        commitment_loss = F.mse_loss(z, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        # Decode
        x_recon = self.decoder(z_q)
        
        return x_recon, vq_loss, encoding_indices.view(x.shape[0], -1)