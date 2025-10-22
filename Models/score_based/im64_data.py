

import torch.nn.functional as F
from torchvision import models

import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from torchvision.models import VGG16_Weights
from typing import List, Optional, Union

class ImageNet64PerceptualDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', img_size: int = 64, 
                 use_vgg_normalization: bool = True, cache_size: int = 10000,
                 selected_classes: Optional[List[int]] = None):
        self.img_size = img_size
        self.use_vgg_normalization = use_vgg_normalization
        self.selected_classes = selected_classes
        self.cache = {} if cache_size > 0 else None
        self.cache_size = cache_size
        
        # VGG16 normalization parameters
        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        self.file_info = []
        self.all_labels = []
        self.all_data_indices = []  # Store original indices for filtering
        self.class_mapping = {}     # Map original class IDs to new indices
        
        if split == 'train':
            file_names = [f"train_data_batch_{i}" for i in range(1, 11)]
        else:
            file_names = ["val_data"]
        
        current_idx = 0
        for file_name in file_names:
            path = os.path.join(root_dir, file_name)
            with open(path, 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
                labels = np.array(batch['labels'], dtype=np.int64)
                data = batch['data']
                
                # Filter by selected classes if specified
                if self.selected_classes is not None:
                    mask = np.isin(labels, self.selected_classes)
                    filtered_labels = labels[mask]
                    filtered_data = data[mask]
                    filtered_indices = np.where(mask)[0] + current_idx
                else:
                    filtered_labels = labels
                    filtered_data = data
                    filtered_indices = np.arange(len(labels)) + current_idx
                
                # Create class mapping if classes are selected
                if self.selected_classes is not None:
                    unique_classes = np.unique(filtered_labels)
                    self.class_mapping.update({cls: i for i, cls in enumerate(unique_classes)})
                
                self.file_info.append({
                    'path': path,
                    'start_idx': current_idx,
                    'end_idx': current_idx + len(labels),
                    'filtered_start': len(self.all_labels),
                    'filtered_end': len(self.all_labels) + len(filtered_labels),
                    'data': filtered_data,
                    'labels': filtered_labels,
                    'original_indices': filtered_indices
                })
                
                self.all_labels.extend(filtered_labels.tolist())
                self.all_data_indices.extend(filtered_indices.tolist())
                current_idx += len(labels)
        
        # Convert to arrays for faster indexing
        self.all_labels = np.array(self.all_labels, dtype=np.int64)
        self.all_data_indices = np.array(self.all_data_indices, dtype=np.int64)
        
        # Remap labels if classes are selected
        if self.selected_classes is not None:
            self.all_labels = np.array([self.class_mapping[cls] for cls in self.all_labels])
    
    def __len__(self):
        return len(self.all_labels)
    
    def get_original_class_distribution(self):
        """Return distribution of original class IDs"""
        if self.selected_classes is None:
            unique, counts = np.unique(self.all_labels, return_counts=True)
            return dict(zip(unique.tolist(), counts.tolist()))
        else:
            # For selected classes, return mapping to original IDs
            reverse_mapping = {v: k for k, v in self.class_mapping.items()}
            remapped_labels = [reverse_mapping[cls] for cls in self.all_labels]
            unique, counts = np.unique(remapped_labels, return_counts=True)
            return dict(zip(unique.tolist(), counts.tolist()))
    
    def _find_file_info(self, idx):
        """Find which file contains the filtered index"""
        for info in self.file_info:
            if info['filtered_start'] <= idx < info['filtered_end']:
                file_idx = idx - info['filtered_start']
                return info, file_idx
        raise IndexError(f"Filtered index {idx} out of range")
    
    def _process_image(self, flat_data):
        H, W = self.img_size, self.img_size
        
        # Reshape from flat array to 3xHxW
        img = np.stack([
            flat_data[0:H*W].reshape(H, W),
            flat_data[H*W:2*H*W].reshape(H, W),
            flat_data[2*H*W:3*H*W].reshape(H, W),
        ], axis=0).astype(np.float32) / 255.0  # [0, 1] range
        
        img = torch.from_numpy(img)
        
        # Apply VGG16 normalization if requested
        if self.use_vgg_normalization:
            img = (img - self.vgg_mean) / self.vgg_std
        
        return img
    
    def __getitem__(self, idx):
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]
        
        info, file_idx = self._find_file_info(idx)
        flat_data = info['data'][file_idx]
        label = self.all_labels[idx]
        
        img = self._process_image(flat_data)
        
        # Cache if enabled
        if self.cache is not None and len(self.cache) < self.cache_size:
            self.cache[idx] = (img.clone(), label)
        
        return img, label


class ImageNet64DataModule:
    def __init__(self, root_dir: str, batch_size: int = 64, num_workers: int = 4, 
                 img_size: int = 64, cache_size: int = 10000,
                 selected_classes: Optional[List[int]] = None):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.cache_size = cache_size
        self.selected_classes = selected_classes
        
        # VGG normalization for perceptual loss
        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def setup(self):
        # Train dataset with augmentations
        self.train_dataset = ImageNet64PerceptualDataset(
            self.root_dir, split='train', img_size=self.img_size,
            use_vgg_normalization=False,  # We'll apply after augmentations
            cache_size=self.cache_size,
            selected_classes=self.selected_classes
        )
        
        # Validation dataset without augmentations
        self.val_dataset = ImageNet64PerceptualDataset(
            self.root_dir, split='val', img_size=self.img_size,
            use_vgg_normalization=True,  # Direct VGG normalization
            cache_size=self.cache_size,
            selected_classes=self.selected_classes
        )
        
        # Print dataset statistics
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Val dataset size: {len(self.val_dataset)}")
        if self.selected_classes is not None:
            print(f"Selected classes: {self.selected_classes}")
            print("Train class distribution:", self.train_dataset.get_original_class_distribution())
            print("Val class distribution:", self.val_dataset.get_original_class_distribution())
    
    def train_dataloader(self):
        # Augmentations for training
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Lambda(lambda x: (x - self.vgg_mean) / self.vgg_std)  # VGG norm after aug
        ])
        
        # Apply transform in collate function
        def collate_fn(batch):
            imgs, labels = zip(*batch)
            imgs = torch.stack([train_transform(img) for img in imgs])
            labels = torch.tensor(labels)
            return imgs, labels
        
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_num_classes(self):
        """Get number of classes (original or filtered)"""
        if self.selected_classes is None:
            return 1000  # Full ImageNet
        else:
            return len(self.selected_classes)
    




class FeaturePerceptualLoss(torch.nn.Module):
    def __init__(self, layers=[3, 8, 15], input_range='0_1'):
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        self.selected_layers = layers
        self.vgg = torch.nn.Sequential(*list(vgg[:max(layers)+1])).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # VGG normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.input_range = input_range

    def preprocess_vgg(self, x):
        """Convert input to VGG16 expected format"""
        if self.input_range == '-1_1':
            # Convert from [-1, 1] to [0, 1]
            x = (x + 1) / 2
        # Normalize with VGG statistics
        return (x - self.mean) / self.std

    def forward(self, recon, target):
        # Preprocess both inputs for VGG
        recon_vgg = self.preprocess_vgg(recon)
        target_vgg = self.preprocess_vgg(target)
        
        loss = 0.0
        x, y = recon_vgg, target_vgg
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.selected_layers:
                loss += F.mse_loss(x, y)
        return loss
    
def vae_loss_with_perceptual(x, xrec, mu, logvar, perc_crit, weight_perc=0.1):
    # Pixel-wise reconstruction
    recon = F.mse_loss(xrec, x) / x.size(0)
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    # Perceptual (feature) loss
    p_loss = perc_crit(xrec, x)
    total = recon + kld + weight_perc * p_loss
    return total, {'recon': recon.item(), 'kld': kld.item(), 'perc': p_loss.item()}