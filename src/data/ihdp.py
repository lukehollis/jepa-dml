import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from pathlib import Path

class IHDPDataset(Dataset):
    """
    IHDP (Infant Health and Development Program) dataset.
    Semi-synthetic dataset with known potential outcomes.
    
    Structure of CSV (30 columns):
    0: Treatment (T)
    1: Factual Outcome (Y)
    2: Counterfactual Outcome (Y_cf)
    3: Mu0 (Noiseless potential outcome under control)
    4: Mu1 (Noiseless potential outcome under treated)
    5-29: Covariates (X) - 25 features
    """
    def __init__(self, root='./datasets/ihdp', split='train', replication=1, train_ratio=0.8, val_ratio=0.1):
        self.root = Path(root)
        self.replication = replication
        
        # Load specific replication
        file_path = self.root / f"ihdp_npci_{replication}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"IHDP file not found: {file_path}")
            
        # Load data
        # No header in these CSVs
        data = pd.read_csv(file_path, header=None).values.astype(np.float32)
        
        self.T = torch.from_numpy(data[:, 0])
        self.Y = torch.from_numpy(data[:, 1])
        self.Y_cf = torch.from_numpy(data[:, 2])
        self.mu0 = torch.from_numpy(data[:, 3])
        self.mu1 = torch.from_numpy(data[:, 4])
        self.X = torch.from_numpy(data[:, 5:])
        
        # Train/Val/Test split
        n_samples = len(data)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        # n_test = remainder
        
        # We use a fixed seed for splitting to ensure consistency across replications if needed,
        # although typically each replication is treated as a full dataset.
        # Standard practice for IHDP is using the provided 1000 replications (files).
        # Usually each file is split into train/val/test.
        
        indices = np.arange(n_samples)
        # To follow CEVAE/standard benchmarks, we might shuffle or take fixed splits.
        # Here we'll do a random shuffle deterministically based on replication ID if we wanted,
        # but typically simple random split is fine.
        
        # Note: Standard IHDP benchmark often uses 63/27/10 splits or 60/30/10.
        # We will strictly respect the requested split argument.
        
        # If split is 'all', return everything (useful for simple evaluation scripts)
        if split == 'all':
            self.indices = indices
        else:
            # Deterministic split based on replication to keep consistent for a given run
            rng = np.random.RandomState(seed=42 + replication)
            rng.shuffle(indices)
            
            if split == 'train':
                self.indices = indices[:n_train]
            elif split == 'val':
                self.indices = indices[n_train:n_train+n_val]
            elif split == 'test':
                self.indices = indices[n_train+n_val:]
            else:
                raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return {
            'X': self.X[real_idx],
            'T': self.T[real_idx],
            'Y': self.Y[real_idx],
            'mu0': self.mu0[real_idx],
            'mu1': self.mu1[real_idx],
            'ATE': (self.mu1 - self.mu0).mean() # Sample ATE
        }

def get_ihdp_loaders(replication=1, batch_size=64, root='./datasets/ihdp'):
    train_set = IHDPDataset(root=root, replication=replication, split='train')
    val_set = IHDPDataset(root=root, replication=replication, split='val')
    test_set = IHDPDataset(root=root, replication=replication, split='test')
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
