import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from src.models.encoder import SpatiotemporalEncoder

class JEPA(nn.Module):
    def __init__(self, encoder, rep_dim):
        super().__init__()
        self.online_encoder = encoder # E_theta
        # Predictor Network (P_phi)
        self.predictor = nn.Sequential(
            nn.Linear(rep_dim, rep_dim * 2),
            nn.GELU(),
            nn.Linear(rep_dim * 2, rep_dim)
        )

    def forward(self, context_views, target_reps):
        context_reps = self.online_encoder(context_views)
        predicted_reps = self.predictor(context_reps)
        # L1 loss as used in V-JEPA 2 reference
        loss = nn.functional.l1_loss(predicted_reps, target_reps)
        return loss

def apply_spatiotemporal_masking(X):
    """
    Placeholder for spatiotemporal masking (e.g., tube masking).
    This defines the context (x) and target (y) views.
    """
    # Simplified: 75% random masking for context, full data for target
    # NOTE: Real applications require structured masking (e.g., masking future frames).
    mask_ratio = 0.75
    mask = torch.rand(X.shape) > mask_ratio
    X_context = X * mask.to(X.device)
    X_target = X
    return X_context, X_target

def train_jepa(train_loader, input_dims, config, device):
    """
    Trains the JEPA model on a specific subset of data (used within DML folds).
    """
    print(f"  Starting JEPA training for {config['jepa_epochs']} epochs...")
    
    # Initialize Encoders (must be initialized fresh for each fold)
    online_encoder = SpatiotemporalEncoder(input_dims, config['rep_dim']).to(device)
    # Target Encoder (E_bar_theta)
    target_encoder = deepcopy(online_encoder).to(device)
    for param in target_encoder.parameters():
        param.requires_grad = False

    model = JEPA(online_encoder, config['rep_dim']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['jepa_lr'])
    
    MOMENTUM = 0.996

    for epoch in range(config['jepa_epochs']):
        epoch_loss = 0
        for batch in train_loader:
            X = batch['X'].to(device)

            # Generate views
            X_context, X_target = apply_spatiotemporal_masking(X)

            # Compute Target Representations (Momentum Encoder, Stop-Gradient)
            with torch.no_grad():
                target_reps = target_encoder(X_target).detach()

            optimizer.zero_grad()
            # Forward pass (Online Encoder and Predictor)
            loss = model(X_context, target_reps)
            loss.backward()
            optimizer.step()

            # Momentum Update (EMA)
            with torch.no_grad():
                for online_param, target_param in zip(online_encoder.parameters(), target_encoder.parameters()):
                    target_param.data = MOMENTUM * target_param.data + (1.0 - MOMENTUM) * online_param.data

            epoch_loss += loss.item()

    # Return the trained Momentum Encoder for downstream representation extraction
    return target_encoder.eval()
