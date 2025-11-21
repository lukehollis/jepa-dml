import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.models.encoder import vit_small_video
from src.models.jepa import apply_tube_masking

class VICReg(nn.Module):
    """
    VICReg implementation (Bardes et al., 2022) adapted for Spatiotemporal data.
    """
    def __init__(self, config):
        super().__init__()
        self.rep_dim = config['rep_dim']
        self.exp_dim = config.get('exp_dim', 2048)
        
        # Encoder
        self.encoder = vit_small_video(
            img_size=config['data_dims'][2],
            patch_size=config.get('patch_size', 8),
            num_frames=config['data_dims'][1],
            tubelet_size=config.get('tubelet_size', 2),
            in_chans=config['data_dims'][0],
            rep_dim=self.rep_dim
        )
        
        # Expander (Projector)
        self.expander = nn.Sequential(
            nn.Linear(self.rep_dim, self.exp_dim),
            nn.BatchNorm1d(self.exp_dim),
            nn.ReLU(),
            nn.Linear(self.exp_dim, self.exp_dim),
            nn.BatchNorm1d(self.exp_dim),
            nn.ReLU(),
            nn.Linear(self.exp_dim, self.exp_dim)
        )

    def forward(self, x):
        rep = self.encoder(x)
        exp = self.expander(rep)
        return rep, exp

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_loss(z1, z2, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
    batch_size = z1.shape[0]
    num_features = z1.shape[1]
    
    # Invariance Loss (MSE)
    repr_loss = F.mse_loss(z1, z2)
    
    # Variance Loss
    std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
    std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_z1)) / 2 + torch.mean(F.relu(1 - std_z2)) / 2
    
    # Covariance Loss
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (batch_size - 1)
    cov_z2 = (z2.T @ z2) / (batch_size - 1)
    cov_loss = off_diagonal(cov_z1).pow(2).sum() / num_features + \
               off_diagonal(cov_z2).pow(2).sum() / num_features
    
    loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
    return loss

def train_vicreg(train_loader, config, device):
    """
    Training loop for VICReg.
    Returns the trained encoder (without expander).
    """
    print(f"Starting VICReg training for {config['jepa_epochs']} epochs...") # Reuse jepa_epochs config
    
    model = VICReg(config).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.get('vicreg_lr', 1e-4), 
        weight_decay=1e-5
    )
    
    for epoch in range(config['jepa_epochs']):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            x = batch['X'].to(device)
            
            # Generate two views: Masked (Context) and Full (Target)
            # Or we could generate two different masks.
            # For simplicity and robustness, let's use Masked vs Full as in JEPA setup
            x_context, x_target, _ = apply_tube_masking(x, mask_ratio=config.get('mask_ratio', 0.75))
            
            optimizer.zero_grad()
            
            _, z1 = model(x_context)
            _, z2 = model(x_target)
            
            loss = vicreg_loss(z1, z2)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{config['jepa_epochs']}, Loss: {epoch_loss/len(train_loader):.4f}")
            
    return model.encoder.eval()
