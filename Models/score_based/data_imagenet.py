import os, pickle
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms


class DownsampledImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=16, transform=None):
        self.img_size = img_size
        self.transform = transform
        data, labels = [], []

        if split == 'train':
            for i in range(1, 11):
                path = os.path.join(root_dir, f"train_data_batch_{i}")
                with open(path, 'rb') as f:
                    batch = pickle.load(f, encoding='latin1')
                data.append(batch['data'])
                labels.extend(batch['labels'])
        else:
            path = os.path.join(root_dir, "val_data")
            with open(path, 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
            data.append(batch['data'])
            labels.extend(batch['labels'])

        self.data = np.concatenate(data, axis=0)           # [N, 3*H*W]
        self.labels = labels                               # List[int]
        self.H = self.W = img_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        flat = self.data[idx]
        img = np.stack([
            flat[0:self.H*self.W].reshape(self.H, self.W),
            flat[self.H*self.W:2*self.H*self.W].reshape(self.H, self.W),
            flat[2*self.H*self.W:3*self.H*self.W].reshape(self.H, self.W),
        ], axis=0)
        img = torch.from_numpy(img.astype(np.float32)) / 255.0
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]




import os
import pickle
import math
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
import random

# reuse _read_batches defined earlier
def _read_batches(root, split='train'):
    data_parts = []
    labels = []
    if split == 'train':
        files = [f"train_data_batch_{i}" for i in range(1, 11)]
    else:
        files = ["val_data"]
    for fn in files:
        path = os.path.join(root, fn)
        with open(path, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
        if 'data' in batch:
            data_parts.append(batch['data'])
        else:
            raise KeyError(f"'data' key not found in {path}")
        if 'labels' in batch:
            labels.extend(batch['labels'])
        elif 'fine_labels' in batch:
            labels.extend(batch['fine_labels'])
        else:
            raise KeyError(f"No 'labels'/'fine_labels' in {path}")
    data = np.concatenate(data_parts, axis=0)  # shape (N, 3*H*W)
    labels = np.array(labels, dtype=np.int64)
    return data, labels


class SubsetDownsampledImageNetDataset(Dataset):
    """
    Dataset supporting selection of a subset of classes.

    Args:
        root_dir: path to folder with pickles (same format as original dataset).
        split: 'train' or other string as used by _read_batches.
        img_size: optional target size (kept for compatibility; actual resizing can be done via `transform`).
        transform: torchvision transform (PIL or tensor-aware depending on return_tensor).
        return_tensor: if True, returns tensors in [0,1] (C,H,W). If False, returns PIL (or transform result).
        selected_classes: None (all classes) or list/iterable of original class ids to keep.
        samples_per_class: None (keep all) or int — maximum number of examples per chosen class (randomly subsampled).
        seed: random seed for reproducible subsampling.
    """
    def __init__(
        self,
        root_dir,
        split='train',
        img_size=None,
        transform=None,
        return_tensor=False,
        selected_classes=None,
        samples_per_class=None,
        seed=0
    ):
        self.transform = transform
        self.return_tensor = return_tensor
        self.target_size = img_size

        data, labels = _read_batches(root_dir, split)  # data: (N, 3*H*W), labels: (N,)
        labels = np.array(labels, dtype=np.int64)

        # If requested, select a subset of classes and remap them to 0..C-1
        self.selected_classes = None
        self.remap = None
        if selected_classes is not None:
            selected_arr = np.array(list(selected_classes), dtype=np.int64)
            # mask where labels are in selected set
            mask = np.isin(labels, selected_arr)
            if not mask.any():
                raise ValueError("No examples found for the provided selected_classes.")
            data = data[mask]
            labels = labels[mask]

            # Remap original class ids to contiguous 0..C-1
            # Keep the order of selected_classes as passed
            self.selected_classes = list(selected_arr)
            self.remap = {int(orig): i for i, orig in enumerate(self.selected_classes)}
            labels = np.array([self.remap[int(l)] for l in labels], dtype=np.int64)
        else:
            # no remapping, keep original labels (but still as numpy array)
            labels = np.array(labels, dtype=np.int64)

        # Optional per-class subsampling
        if samples_per_class is not None:
            if samples_per_class <= 0:
                raise ValueError("samples_per_class must be > 0 or None.")
            rng = np.random.RandomState(seed)
            kept_idx = []
            unique_classes = np.unique(labels)
            for c in unique_classes:
                idxs = np.nonzero(labels == c)[0]
                if len(idxs) <= samples_per_class:
                    kept = idxs
                else:
                    kept = rng.choice(idxs, size=samples_per_class, replace=False)
                kept_idx.append(kept)
            kept_idx = np.concatenate(kept_idx)
            # sort to keep some deterministic order
            kept_idx = np.sort(kept_idx)
            data = data[kept_idx]
            labels = labels[kept_idx]

        # final storage
        self.raw = data  # shape (N_filtered, 3*H*W)
        self.labels = labels.astype(np.int64)

        # infer original H,W from vector length (safe)
        veclen = self.raw.shape[1]
        per_ch = veclen // 3
        orig_side = int(math.sqrt(per_ch))
        if orig_side * orig_side != per_ch:
            raise ValueError("Cannot infer H,W from data vector length")
        self.orig_H = self.orig_W = orig_side

    def __len__(self):
        return len(self.labels)

    def _flat_to_pil(self, flat):
        H, W = self.orig_H, self.orig_W
        ch0 = flat[0:H*W].reshape(H, W)
        ch1 = flat[H*W:2*H*W].reshape(H, W)
        ch2 = flat[2*H*W:3*H*W].reshape(H, W)
        img = np.stack([ch0, ch1, ch2], axis=2)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def __getitem__(self, idx):
        flat = self.raw[idx]
        label = int(self.labels[idx])

        if self.return_tensor:
            H, W = self.orig_H, self.orig_W
            img = np.stack([
                flat[0:H*W].reshape(H, W),
                flat[H*W:2*H*W].reshape(H, W),
                flat[2*H*W:3*H*W].reshape(H, W)
            ], axis=0).astype(np.float32) / 255.0  # C,H,W in [0,1]
            img = torch.from_numpy(img)
            if self.transform:
                img = self.transform(img)  # transform must accept tensor
            return img, label
        else:
            pil = self._flat_to_pil(flat)
            if self.transform:
                pil = self.transform(pil)
            else:
                pil = T.ToTensor()(pil)
            return pil, label



