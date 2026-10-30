"""One manifest for all models; noise and crop positions are deterministic."""
from __future__ import annotations
from pathlib import Path
import csv
import hashlib
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset


def seed_for(filename: str, epoch: int = 0, seed: int = 42) -> int:
    b = f'{seed}|{filename}|{epoch}'.encode('utf8')
    return int.from_bytes(hashlib.sha256(b).digest()[:8], 'big') % (2**32)


def add_noise(clean: np.ndarray, sigma=25, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(clean.astype(np.float32) + rng.normal(0, sigma / 255., clean.shape).astype(np.float32), 0., 1.)


class DenoiseDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str, patch=128, sigma=25, seed=42, epoch=0):
        self.manifest = Path(manifest).resolve()
        with self.manifest.open(newline='', encoding='utf-8') as f:
            self.files = [row['path'] for row in csv.DictReader(f) if row['split'] == split]
        if not self.files:
            raise ValueError(f'No files in split {split}. Prepare the dataset first.')
        self.split = split
        self.patch = patch
        self.sigma = sigma
        self.seed = seed
        self.epoch = epoch

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        entry = self.files[i]
        f = (self.manifest.parent / entry).resolve()
        if not f.is_relative_to(self.manifest.parent):
            raise ValueError('Manifest contains a path outside its directory')
        with Image.open(f) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            # Center image at a minimum patch size to avoid variable shapes.
            if min(im.size) < self.patch:
                scale = self.patch / min(im.size)
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.BICUBIC)
            if self.split == 'train':
                rng = np.random.default_rng(seed_for(entry, self.epoch, self.seed))
                x = int(rng.integers(0, im.width - self.patch + 1))
                y = int(rng.integers(0, im.height - self.patch + 1))
            else:
                x = (im.width - self.patch) // 2
                y = (im.height - self.patch) // 2
            im = im.crop((x, y, x + self.patch, y + self.patch))
            clean = np.asarray(im, dtype=np.float32) / 255.
        noise_seed = seed_for(entry, self.epoch if self.split == 'train' else 0, self.seed + 11)
        noisy = add_noise(clean, sigma=self.sigma, seed=noise_seed)
        return (torch.from_numpy(noisy.copy()).permute(2, 0, 1),
                torch.from_numpy(clean.copy()).permute(2, 0, 1))
