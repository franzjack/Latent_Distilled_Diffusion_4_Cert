
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights

class Encoder16(nn.Module):
    def __init__(self, z_dim=16, base=64, img_size=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, base, 3, stride=2, padding=1),      # 16 → 8
            nn.ReLU(True),
            nn.Conv2d(base, base * 2, 3, 2, 1),               # 8 → 4
            nn.ReLU(True)
        )
        spatial = img_size // 4  # = 4
        feat_dim = base * 2 * spatial * spatial  # base*2 * (4*4) = base*2 * 16
        self.fc_mu = nn.Linear(feat_dim, z_dim)
        self.fc_lv = nn.Linear(feat_dim, z_dim)

    def forward(self, x):
        h = self.conv(x)               # → [B, base*2, 4, 4]
        h = h.view(h.size(0), -1)      # flatten
        mu, logvar = self.fc_mu(h), self.fc_lv(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return z, mu, logvar


class Decoder16(nn.Module):
    def __init__(self, z_dim=16, base=64, img_size=16):
        super().__init__()
        spatial = img_size // 4  # = 4
        feat_dim = base * 2 * spatial * spatial
        self.fc = nn.Linear(z_dim, feat_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, kernel_size=4, stride=2, padding=1),  # 4 → 8
            nn.ReLU(True),
            nn.ConvTranspose2d(base, 3, kernel_size=4, stride=2, padding=1),        # 8 → 16
            nn.Sigmoid()
        )

    def forward(self, z):
        h = F.relu(self.fc(z))         # → [B, feat_dim]
        B = z.size(0)
        h = h.view(B, -1, 4, 4)        # reshape to [B, base*2, 4, 4]
        xrec = self.deconv(h)          # → [B, 3, 16, 16]
        return xrec


class VAE16(nn.Module):
    def __init__(self, img_size=16, z_dim=16, base=64):
        super().__init__()
        self.encoder = Encoder16(z_dim=z_dim, base=base, img_size=img_size)
        self.decoder = Decoder16(z_dim=z_dim, base=base, img_size=img_size)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        xrec = self.decoder(z)
        return xrec, mu, logvar
    


# -------------------------------------------------
# Residual Block (fixed-shape, CIFAR10-friendly)
# -------------------------------------------------
class ResBlockCIFAR(nn.Module):
    """
    Residual block for CIFAR-10 (32x32).
    - Keeps spatial size constant
    - Expects in_ch == out_ch
    - No if-clauses, always safe to add skip
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


# -------------------------------------------------
# Encoder
# -------------------------------------------------
class Encoder32(nn.Module):
    def __init__(self, z_dim=128, base=64):
        super().__init__()
        # Stem: reduce resolution while expanding channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
        )
        # Residual blocks at bottleneck (8x8)
        self.res1 = ResBlockCIFAR(base * 2)
        self.res2 = ResBlockCIFAR(base * 2)

        # Latent projection
        feat_dim = (base * 2) * 8 * 8
        self.fc_mu = nn.Linear(feat_dim, z_dim)
        self.fc_logvar = nn.Linear(feat_dim, z_dim)

    def forward(self, x):
        h = self.stem(x)
        h = self.res1(h)
        h = self.res2(h)
        h = torch.flatten(h, 1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return z, mu, logvar


# -------------------------------------------------
# Decoder
# -------------------------------------------------
class Decoder32(nn.Module):
    def __init__(self, z_dim=128, base=64):
        super().__init__()
        self.fc = nn.Linear(z_dim, (base * 2) * 8 * 8)

        self.res1 = ResBlockCIFAR(base * 2)
        self.res2 = ResBlockCIFAR(base * 2)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, 3, 4, stride=2, padding=1),  # 16 -> 32
            nn.Sigmoid(),  # outputs in [0,1]
        )

    def forward(self, z):
        h = self.fc(z)
        B = z.size(0)
        h = h.view(B, -1, 8, 8)
        h = self.res1(h)
        h = self.res2(h)
        return self.deconv(h)


# -------------------------------------------------
# VAE Wrapper
# -------------------------------------------------
class VAE32(nn.Module):
    def __init__(self, z_dim=128, base=64):
        super().__init__()
        self.encoder = Encoder32(z_dim=z_dim, base=base)
        self.decoder = Decoder32(z_dim=z_dim, base=base)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        xrec = self.decoder(z)
        return xrec, mu, logvar

    


class Encoder64(nn.Module):
    def __init__(self, z_dim=128, base=64, img_size=64):
        """
        Encoder that maps 64x64 -> latent z (vector).
        By default z_dim=128 (you can reduce to e.g. 64).
        """
        super().__init__()
        # conv blocks: 64->32->16->8->4 (4 halving steps)
        self.conv = nn.Sequential(
            nn.Conv2d(3, base, 4, 2, 1),    # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base*2, 4, 2, 1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*4, 4, 2, 1),# 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(base*4, base*8, 4, 2, 1),# 8 -> 4
            nn.ReLU(inplace=True),
        )
        spatial = img_size // (2**4)  # 64//16 = 4
        feat_dim = base * 8 * spatial * spatial
        self.fc_mu = nn.Linear(feat_dim, z_dim)
        self.fc_logvar = nn.Linear(feat_dim, z_dim)

    def forward(self, x):
        h = self.conv(x)                   # [B, base*8, 4, 4]
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class Decoder64(nn.Module):
    def __init__(self, z_dim=128, base=64, img_size=64):
        """
        Decoder that maps vector z -> 64x64 RGB.
        Uses ConvTranspose2d; avoids BatchNorm to be friendlier for verification.
        """
        super().__init__()
        spatial = img_size // (2**4)  # 4
        feat_dim = base * 8 * spatial * spatial
        self.fc = nn.Linear(z_dim, feat_dim)
        self.deconv = nn.Sequential(
            # input shape after view: [B, base*8, 4, 4]
            nn.ConvTranspose2d(base*8, base*4, 4, 2, 1),  # 4 -> 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base*4, base*2, 4, 2, 1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base*2, base,   4, 2, 1),  # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, 3, 4, 2, 1),         # 32 -> 64
            nn.Sigmoid()
        )

    def forward(self, z):
        h = F.relu(self.fc(z))
        B = z.size(0)
        # reshape to [B, base*8, 4, 4]
        # determine channel dim programmatically for safety
        ch = (self.fc.out_features) // (4*4)
        h = h.view(B, ch, 4, 4)
        xrec = self.deconv(h)
        return xrec


class VAE64(nn.Module):
    def __init__(self, img_size=64, z_dim=128, base=64):
        super().__init__()
        assert img_size == 64, "This class targets 64x64 images"
        self.encoder = Encoder64(z_dim=z_dim, base=base, img_size=img_size)
        self.decoder = Decoder64(z_dim=z_dim, base=base, img_size=img_size)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        xrec = self.decoder(z)
        return xrec, mu, logvar


# ---- VAE loss (fixed scaling) ----


# ---- Perceptual loss based on VGG (safe normalization + early-layer advice) ----
class FeaturePerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    - inputs (recon, target) must be in [0,1], shape [B,3,H,W].
    - This class will internally normalize using ImageNet mean/std.
    - Choose layers so they are compatible with H,W. For 64x64 default layers are OK.
    """
    def __init__(self, layers=[3, 8, 15], device=None):
        super().__init__()
        # load VGG features with weights
        full_vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        max_layer = max(layers)
        # keep features up to max selected layer (indexing matches original features)
        self.vgg = nn.Sequential(*list(full_vgg[: max_layer + 1])).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        # selected layer indices relative to the sliced vgg (they are the same indices)
        self.selected_layers = set(layers)

        # ImageNet normalization (tensors will be broadcasted)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if device is not None:
            self.to(device)

    def forward(self, recon, target):
        """
        recon, target: [B,3,H,W] in [0,1] (float)
        Returns MSE over selected feature maps (sum of layer-wise mean squared differences).
        """
        # safety: ensure float and same device
        assert recon.dtype.is_floating_point, "recon must be float tensor in [0,1]"
        assert target.dtype.is_floating_point, "target must be float tensor in [0,1]"

        # normalize to ImageNet statistics expected by VGG
        mean = self.mean.to(recon.device)
        std = self.std.to(recon.device)
        x = (recon - mean) / std
        y = (target - mean) / std

        loss = 0.0
        xi, yi = x, y
        # iterate over sliced vgg layers; compare feature activations at selected indices
        for idx, layer in enumerate(self.vgg):
            xi = layer(xi)
            yi = layer(yi)
            if idx in self.selected_layers:
                # use mean MSE over elements in this feature map
                loss += F.mse_loss(xi, yi, reduction='mean')

        return loss


# ---- VAE loss with perceptual (fixed recon scaling) ----
def vae_loss(x, xrec, mu, logvar):
    recon = F.mse_loss(xrec, x, reduction='sum') / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total = recon + kld
    return total, {'recon': float(recon), 'kld': float(kld)}

def vae_loss_with_perceptual(x, xrec, mu, logvar, perc_crit, weight_perc=0.1):
    recon = F.mse_loss(xrec, x, reduction='sum') / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    perc = perc_crit(xrec, x)
    total = recon + kld + weight_perc * perc
    return total, {'recon': float(recon), 'kld': float(kld), 'perc': float(perc)}


