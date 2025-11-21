import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from pathlib import Path

class TwinsDataset(Dataset):
    """
    Twins dataset for causal inference.
    Observational dataset based on twin births in the USA.
    Treatment: Born heavier (T=1) vs lighter (T=0)
    Outcome: Mortality (binary)
    
    If data is missing, allows synthetic generation for testing pipeline.
    """
    def __init__(self, root='./datasets/twins', split='train', train_ratio=0.8, val_ratio=0.1, synthetic=False):
        self.root = Path(root)
        self.synthetic = synthetic
        
        if synthetic:
            self._generate_synthetic()
        else:
            self._load_real_data()
            
        # Train/Val/Test split
        n_samples = len(self.T)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        indices = np.arange(n_samples)
        # Fixed seed for consistency
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        
        if split == 'train':
            self.indices = indices[:n_train]
        elif split == 'val':
            self.indices = indices[n_train:n_train+n_val]
        elif split == 'test':
            self.indices = indices[n_train+n_val:]
        else:
            self.indices = indices # 'all'

    def _load_real_data(self):
        # This expects the standard processed Twins dataset
        # Usually 'twins_X.csv', 'twins_T.csv', 'twins_Y.csv' or similar
        # Or a single file.
        # Since we don't have it, we'll check for a common format or raise error
        
        data_path = self.root / "twins.csv"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Twins dataset not found at {data_path}. "
                "Please download it manually or use synthetic=True for testing."
            )
            
        df = pd.read_csv(data_path)
        
        # Assuming standard columns if available
        # We might need to adjust this based on the actual file format provided by user
        if 'T' in df.columns and 'Y' in df.columns:
            self.T = torch.from_numpy(df['T'].values).float()
            self.Y = torch.from_numpy(df['Y'].values).float()
            # Assume all other columns are X
            x_cols = [c for c in df.columns if c not in ['T', 'Y', 'Y0', 'Y1']]
            self.X = torch.from_numpy(df[x_cols].values).float()
            
            if 'Y0' in df.columns and 'Y1' in df.columns:
                self.mu0 = torch.from_numpy(df['Y0'].values).float()
                self.mu1 = torch.from_numpy(df['Y1'].values).float()
            else:
                # In observational data we don't always have potential outcomes
                # But for Twins benchmark, we often use the other twin as counterfactual
                self.mu0 = torch.zeros_like(self.Y) # Placeholder
                self.mu1 = torch.zeros_like(self.Y) # Placeholder
        else:
            # Fallback or error
            raise ValueError("Unknown Twins CSV format. Expected columns T, Y.")

    def _generate_synthetic(self):
        print("Generating synthetic Twins-like data for testing...")
        n_samples = 5000
        n_features = 30
        
        # Generate covariates
        self.X = torch.randn(n_samples, n_features)
        
        # Generate treatment (propensity depends on X)
        logits = self.X[:, 0] + 0.5 * self.X[:, 1]
        prob = torch.sigmoid(logits)
        self.T = torch.bernoulli(prob)
        
        # Generate outcome
        # Y = T + X + noise
        noise = torch.randn(n_samples) * 0.1
        y0 = self.X[:, 2] + 0.5 * self.X[:, 3]
        y1 = y0 + 1.0 # ATE = 1.0
        
        self.mu0 = y0
        self.mu1 = y1
        
        self.Y = torch.where(self.T == 1, y1, y0) + noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return {
            'X': self.X[real_idx],
            'T': self.T[real_idx],
            'Y': self.Y[real_idx],
            'mu0': self.mu0[real_idx],
            'mu1': self.mu1[real_idx]
        }

def get_twins_loaders(batch_size=64, root='./datasets/twins', synthetic=False):
    # Check if real data exists, otherwise default to synthetic if allowed (or error)
    # Ideally we want to be explicit.
    
    train_set = TwinsDataset(root=root, split='train', synthetic=synthetic)
    val_set = TwinsDataset(root=root, split='val', synthetic=synthetic)
    test_set = TwinsDataset(root=root, split='test', synthetic=synthetic)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
