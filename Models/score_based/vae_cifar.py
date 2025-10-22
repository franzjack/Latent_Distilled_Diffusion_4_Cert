# train_vae32_fixed.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms, utils as tv_utils
import matplotlib.pyplot as plt
from tqdm import tqdm

# If you use VGG-based perceptual loss:
from torchvision.models import VGG16_Weights
from torchvision import models

# -------------------------------
# Utilities & data loader helper
# -------------------------------
def get_data_loader(dataset, batch_size, cuda=False, num_workers=None, pin_memory=True, shuffle=True):
    if num_workers is None:
        num_workers = 4 if cuda else 0
    kwargs = {}
    if cuda:
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

def safe_pil_to_tensor_no_numpy(pic):
    # pic is a PIL image
    data = list(pic.getdata())  # Python-level, slow but safe
    t = torch.tensor(data, dtype=torch.uint8)
    t = t.view(pic.size[1], pic.size[0], 3)  # H, W, C
    return t.permute(2, 0, 1).float().div(255.0)

def warn_if_normalization_mismatch(x):
    # x is a batch tensor on cpu or cuda
    low = float(x.min().item())
    high = float(x.max().item())
    if low < 0.0 - 1e-3 or high > 1.0 + 1e-3:
        print(f"WARNING: input images appear outside [0,1] (min={low:.3f}, max={high:.3f}). "
              "Make sure dataset transforms and model outputs use the same range.")
    else:
        # ok
        pass


# -------------------------------
# Residual block and VAE model
# -------------------------------
class ResBlockCIFAR(nn.Module):
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


class Encoder32(nn.Module):
    def __init__(self, z_dim=128, base=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResBlockCIFAR(base * 2)
        self.res2 = ResBlockCIFAR(base * 2)
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


class VAE32(nn.Module):
    def __init__(self, z_dim=128, base=64):
        super().__init__()
        self.encoder = Encoder32(z_dim=z_dim, base=base)
        self.decoder = Decoder32(z_dim=z_dim, base=base)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        xrec = self.decoder(z)
        return xrec, mu, logvar


# -------------------------------
# Perceptual loss (VGG)
# -------------------------------
class FeaturePerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8], device=None):
        """
        Uses selected VGG16 feature indices. Default uses shallow layers safe for 32x32.
        Inputs must be in [0,1], shape [B,3,H,W]. This module internally normalizes to ImageNet mean/std.
        """
        super().__init__()
        full_vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        max_layer = max(layers)
        self.vgg = nn.Sequential(*list(full_vgg[: max_layer + 1])).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.selected_layers = set(layers)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if device is not None:
            self.to(device)

    def forward(self, recon, target):
        # recon, target: [B,3,H,W] in [0,1]
        mean = self.mean.to(recon.device)
        std = self.std.to(recon.device)
        x = (recon - mean) / std
        y = (target - mean) / std
        loss = 0.0
        xi, yi = x, y
        for idx, layer in enumerate(self.vgg):
            xi = layer(xi)
            yi = layer(yi)
            if idx in self.selected_layers:
                loss = loss + F.mse_loss(xi, yi, reduction='mean')
        return loss


# -------------------------------
# Loss helpers (ELBO scaling)
# -------------------------------
def loss_components(x, xrec, mu, logvar, perc_crit=None):
    """
    Returns recon (per-sample average), kld (per-sample average), perc (scalar or 0)
    recon: sum over pixels per image, then averaged over batch (=ELBO convention)
    kld: sum over latent dims per image, then averaged over batch
    """
    # reconstruction: sum over all pixels then divide by batch (ELBO-style)
    recon = F.mse_loss(xrec, x, reduction='sum') / x.size(0)  # scalar tensor
    # KL: sum over latent dims then average over batch
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    perc = torch.tensor(0.0, device=x.device)
    if perc_crit is not None:
        perc = perc_crit(xrec, x)
    return recon, kld, perc


# -------------------------------
# Visualization helper
# -------------------------------
def show_batch_reconstructions(model, dataloader, device='cuda', n=8, foldername=""):
    model.eval()
    imgs, _ = next(iter(dataloader))
    imgs = imgs[:n].to(device)
    with torch.no_grad():
        xrec, _, _ = model(imgs)
    # create grid: originals on top, reconstructions below
    grid = tv_utils.make_grid(torch.cat([imgs.cpu(), xrec.cpu()], dim=0), nrow=n, normalize=False, padding=2)
    plt.figure(figsize=(n*1.5, 3))
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0).clamp(0,1).numpy())
    if foldername:
        os.makedirs(foldername, exist_ok=True)
        plt.savefig(os.path.join(foldername, "recon_grid.png"))
    plt.show()
    plt.close()


# -------------------------------
# Training function (corrected)
# -------------------------------
def train(
    model,
    train_loader,
    *,
    epochs=50,
    lr=2e-4,
    foldername="",
    device='cuda',
    perceptual=False,
    kl_anneal_epochs=20,
    weight_perc=0.05,
    free_bits=None,   # e.g., 0.5 nats per-dim, or None to disable
    use_amp=False
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    opt = Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    # create perceptual once if requested
    perc_crit = FeaturePerceptualLoss(device=device, layers=[3, 8]).to(device) if perceptual else None

    for ep in range(1, epochs + 1):
        model.train()
        running_recon = 0.0
        running_kld = 0.0
        running_perc = 0.0
        running_loss = 0.0

        kl_weight = min(1.0, ep / float(max(1, kl_anneal_epochs)))

        for batch_idx, (x, _) in enumerate(tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", leave=False)):
            x = x.to(device)
            # quick check for normalization mismatch on first batch
            if batch_idx == 0 and ep == 1:
                warn_if_normalization_mismatch(x)

            opt.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    xrec, mu, logvar = model(x)
                    recon, kld, perc = loss_components(x, xrec, mu, logvar, perc_crit)
                    # apply free-bits if requested (per-dim)
                    if free_bits is not None:
                        kld_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)  # [B, z]
                        kld_per_sample = torch.sum(torch.clamp(kld_per_dim - free_bits, min=0.0), dim=1).mean()
                        # note: we replaced kld with free-bits version (already averaged)
                        kld = kld_per_sample
                    total_loss = recon + kl_weight * kld + (weight_perc * perc if perceptual else 0.0)
                scaler.scale(total_loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                xrec, mu, logvar = model(x)
                recon, kld, perc = loss_components(x, xrec, mu, logvar, perc_crit)
                if free_bits is not None:
                    kld_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
                    kld_per_sample = torch.sum(torch.clamp(kld_per_dim - free_bits, min=0.0), dim=1).mean()
                    kld = kld_per_sample
                total_loss = recon + kl_weight * kld + (weight_perc * perc if perceptual else 0.0)
                total_loss.backward()
                opt.step()

            running_recon += float(recon.detach().cpu().item())
            running_kld += float(kld.detach().cpu().item())
            running_perc += float(perc.detach().cpu().item()) if perceptual else 0.0
            running_loss += float(total_loss.detach().cpu().item())

        n_batches = len(train_loader)
        print(f"Epoch {ep:03d} | loss={running_loss / n_batches:.4f} | recon={running_recon / n_batches:.4f} | "
              f"kld={running_kld / n_batches:.4f} | kl_w={kl_weight:.3f} | perc={running_perc / n_batches:.6f}")

        # checkpoint & visualizations
        if foldername and ep % 10 == 0:
            os.makedirs(foldername, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(foldername, f"vae_epoch{ep}.pth"))
            show_batch_reconstructions(model, train_loader, device=device, n=8, foldername=foldername)

    # final save
    if foldername:
        os.makedirs(foldername, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(foldername, "vae_final.pth"))

    return model
