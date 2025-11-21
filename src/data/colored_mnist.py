import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
import os
from PIL import Image

class ColoredMNISTDataset(Dataset):
    """
    Colored MNIST dataset for causal inference.
    
    Treatment (T): Digit identity (Binary: < 5 vs >= 5)
    Confounder (Z): Color (Binary: Red vs Green)
    Outcome (Y): Synthetic function of T and Z
    
    The color is highly correlated with the digit label (confounding).
    """
    def __init__(self, root='./datasets', train=True, download=True, 
                 confounding_strength=0.9, noise_level=0.1):
        self.root = root
        self.train = train
        self.confounding_strength = confounding_strength
        
        # Load standard MNIST
        self.mnist = datasets.MNIST(
            root=root, 
            train=train, 
            download=download,
            transform=None # We process manually
        )
        
        self.data = self._generate_data()

    def _generate_data(self):
        images = self.mnist.data
        targets = self.mnist.targets
        
        n_samples = len(images)
        
        # 1. Define Treatment T based on Digit
        # T = 0 if digit < 5, T = 1 if digit >= 5
        T = (targets >= 5).float()
        
        # 2. Generate Confounder Z (Color) correlated with T
        # Z = T with probability 'confounding_strength', else 1-T
        # Z=0: Red, Z=1: Green
        coin_flips = torch.bernoulli(torch.full((n_samples,), self.confounding_strength))
        Z = torch.where(coin_flips == 1, T, 1 - T)
        
        # 3. Color the images based on Z
        colored_images = []
        for i in range(n_samples):
            img = images[i]
            z_val = Z[i].item()
            
            # Create RGB image
            # MNIST is (28, 28) -> (28, 28, 3)
            img_rgb = torch.stack([img, img, img], dim=0) # (3, H, W)
            
            # Apply color
            if z_val == 0: # Red
                img_rgb[1, :, :] = 0 # Zero out Green
                img_rgb[2, :, :] = 0 # Zero out Blue
            else: # Green
                img_rgb[0, :, :] = 0 # Zero out Red
                img_rgb[2, :, :] = 0 # Zero out Blue
                
            colored_images.append(img_rgb)
            
        X = torch.stack(colored_images).float() / 255.0
        
        # 4. Generate Outcome Y
        # Simple structural equation: Y = 2*T + 3*Z + noise
        # This means both the digit and the color affect the outcome.
        # Since Z is correlated with T, simply regressing Y on T will be biased.
        noise = torch.randn(n_samples) * 0.5
        Y = 2.0 * T + 3.0 * Z + noise
        
        # True ATE = 2.0 (Direct effect of T)
        
        return {
            'X': X,
            'T': T,
            'Y': Y.float(),
            'Z': Z, # Confounder (ground truth, usually unobserved in real settings)
            'digits': targets
        }

    def __len__(self):
        return len(self.data['X'])

    def __getitem__(self, idx):
        return {
            'X': self.data['X'][idx],
            'T': self.data['T'][idx],
            'Y': self.data['Y'][idx],
            'Z': self.data['Z'][idx]
        }

def get_colored_mnist_loaders(batch_size=64, **kwargs):
    train_set = ColoredMNISTDataset(train=True, **kwargs)
    test_set = ColoredMNISTDataset(train=False, **kwargs)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
