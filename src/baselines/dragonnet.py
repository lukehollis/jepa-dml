import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.models.encoder import vit_small_video

class DragonNet(nn.Module):
    """
    DragonNet implementation (Shi et al., 2019) adapted for Spatiotemporal data.
    Uses a ViT encoder for feature extraction followed by DragonNet heads.
    """
    def __init__(self, config):
        super().__init__()
        self.rep_dim = config['rep_dim']
        
        # 1. Shared Representation Learning (Encoder)
        # Reusing the same backbone structure as JEPA for fair comparison
        self.encoder = vit_small_video(
            img_size=config['data_dims'][2],
            patch_size=config.get('patch_size', 8),
            num_frames=config['data_dims'][1],
            tubelet_size=config.get('tubelet_size', 2),
            in_chans=config['data_dims'][0],
            rep_dim=self.rep_dim
        )

        # 2. DragonNet Heads
        # Propensity Head (g)
        self.propensity_head = nn.Sequential(
            nn.Linear(self.rep_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Outcome Head 0 (Q0) - for T=0
        self.outcome_head_0 = nn.Sequential(
            nn.Linear(self.rep_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
        
        # Outcome Head 1 (Q1) - for T=1
        self.outcome_head_1 = nn.Sequential(
            nn.Linear(self.rep_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
        
        # Targeted Regularization Epsilon
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Shared Representation
        z = self.encoder(x)
        
        # Propensity Score
        g = self.propensity_head(z)
        
        # Conditional Outcomes
        y0 = self.outcome_head_0(z)
        y1 = self.outcome_head_1(z)
        
        return y0, y1, g, self.epsilon

def targeted_regularization_loss(y_true, t_true, y0_pred, y1_pred, g_pred, epsilon, alpha=1.0, beta=1.0):
    """
    Computes DragonNet loss with Targeted Regularization.
    L = L_y + alpha * L_g + beta * L_treg
    """
    # 1. Outcome Loss (L_y)
    # Select the predicted outcome corresponding to the true treatment
    y_pred = t_true * y1_pred + (1 - t_true) * y0_pred
    loss_y = nn.functional.mse_loss(y_pred, y_true)
    
    # 2. Propensity Loss (L_g)
    loss_g = nn.functional.binary_cross_entropy(g_pred, t_true)
    
    # 3. Targeted Regularization Loss (L_treg)
    # Calculate the targeted perturbation
    # h = (T - g) / (g * (1 - g))
    # For numerical stability, add a small epsilon to denominator
    g_pred_stable = torch.clamp(g_pred, 0.01, 0.99)
    h = (t_true - g_pred_stable) / (g_pred_stable * (1 - g_pred_stable))
    
    # Targeted prediction
    y_pred_targeted = y_pred + epsilon * h
    loss_treg = nn.functional.mse_loss(y_pred_targeted, y_true)
    
    # Total Loss
    total_loss = loss_y + alpha * loss_g + beta * loss_treg
    
    return total_loss, loss_y, loss_g, loss_treg

def train_dragonnet(train_loader, val_loader, config, device):
    """
    Training loop for DragonNet.
    """
    model = DragonNet(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-4), weight_decay=1e-5)
    
    print(f"Starting DragonNet training for {config['epochs']} epochs...")
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            x = batch['X'].to(device)
            t = batch['T'].to(device).view(-1, 1)
            y = batch['Y'].to(device).view(-1, 1)
            
            optimizer.zero_grad()
            
            y0, y1, g, eps = model(x)
            
            loss, l_y, l_g, l_treg = targeted_regularization_loss(
                y, t, y0, y1, g, eps,
                alpha=config.get('dragonnet_alpha', 1.0),
                beta=config.get('dragonnet_beta', 1.0)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}, Loss: {train_loss/len(train_loader):.4f}")

    return model

def predict_dragonnet(model, loader, device):
    """
    Inference using DragonNet. Returns ATE estimate.
    """
    model.eval()
    y1_preds = []
    y0_preds = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch['X'].to(device)
            y0, y1, g, eps = model(x)
            
            # Apply targeted regularization adjustment for final prediction?
            # Standard DragonNet usage often uses the base outputs, 
            # but t-reg implies using the updated ones. 
            # However, for ATE = E[Y1 - Y0], the perturbation might cancel out or be small.
            # We'll return the raw head predictions as is standard for the ATE calculation.
            
            y1_preds.append(y1.cpu())
            y0_preds.append(y0.cpu())
            
    y1_all = torch.cat(y1_preds)
    y0_all = torch.cat(y0_preds)
    
    # ATE = mean(Y1 - Y0)
    ate = (y1_all - y0_all).mean().item()
    return ate
