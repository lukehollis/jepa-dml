"""
Joint Embedding Predictive Architecture (JEPA) for spatiotemporal video.
Implements V-JEPA 2 training with tube masking and momentum encoder.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from src.models.encoder import vit_small_video, vit_base_video
from src.models.masking import generate_tube_mask


class JEPA(nn.Module):
    """
    JEPA model with online encoder and predictor.
    Target encoder is separate (momentum-updated).
    """
    def __init__(self, encoder, rep_dim):
        super().__init__()
        self.online_encoder = encoder  # E_theta
        
        # Predictor Network (P_phi) - projects context to target space
        self.predictor = nn.Sequential(
            nn.Linear(rep_dim, rep_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(rep_dim * 2, rep_dim)
        )

    def forward(self, x_full, compute_target=False):
        """
        Forward pass through encoder and predictor.
        
        Args:
            x_full: Full video tensor
            compute_target: If True, return encoder output directly (for target)
        
        Returns:
            Predicted or encoded representations
        """
        reps = self.online_encoder(x_full)
        
        if compute_target:
            return reps  # Don't apply predictor for target
        
        # Apply predictor for context
        predicted_reps = self.predictor(reps)
        return predicted_reps


def apply_tube_masking(X, mask_ratio=0.75, num_masks=4):
    """
    Apply spatiotemporal tube masking to video batch.
    
    Args:
        X: (B, C, T, H, W) video tensor
        mask_ratio: Target fraction of patches to mask
        num_masks: Number of tube blocks to generate
    
    Returns:
        X_context: Masked video (for context encoder)
        X_target: Full video (for target encoder)
        masks: Boolean mask tensors
    """
    B, C, T, H, W = X.shape
    
    # Assume patch_size=8, tubelet_size=2 (from config)
    patch_size = 8
    tubelet_size = 2
    
    H_patches = H // patch_size
    W_patches = W // patch_size
    num_patches_per_frame = H_patches * W_patches
    
    # Generate tube masks
    masks = generate_tube_mask(
        batch_size=B,
        num_frames=T,
        num_patches_per_frame=num_patches_per_frame,
        mask_ratio=mask_ratio,
        tubelet_size=tubelet_size,
        num_masks=num_masks
    ).to(X.device)
    
    # Apply masking by zeroing out patches
    # Reshape mask to (B, T//tubelet, H_patches, W_patches)
    T_tubelets = T // tubelet_size
    mask_spatial = masks.reshape(B, T_tubelets, H_patches, W_patches)
    
    # Expand to full resolution
    mask_full = mask_spatial.unsqueeze(1)  # (B, 1, T_tubelets, H_patches, W_patches)
    mask_full = mask_full.repeat_interleave(tubelet_size, dim=2)  # Expand temporally
    mask_full = mask_full.repeat_interleave(patch_size, dim=3)  # Expand spatially H
    mask_full = mask_full.repeat_interleave(patch_size, dim=4)  # Expand spatially W
    mask_full = mask_full.repeat(1, C, 1, 1, 1)  # Add channel dim
    
    # Create context (masked) and target (full)
    X_context = X * (~mask_full).float()  # Mask out (zero) the masked regions
    X_target = X  # Keep full video for target
    
    return X_context, X_target, masks


def train_jepa(train_loader, config, device):
    """
    Train JEPA model with tube masking and momentum encoder.
    This is called once per DML fold.
    
    Args:
        train_loader: DataLoader for training data
        config: Configuration dict
        device: torch device
    
    Returns:
        target_encoder: Trained momentum encoder for representation extraction
    """
    print(f"  Starting JEPA training for {config['jepa_epochs']} epochs...")
    
    # Initialize online encoder
    if config.get('vit_size', 'small') == 'small':
        online_encoder = vit_small_video(
            img_size=config['data_dims'][2],  # H (assumes square)
            patch_size=config.get('patch_size', 8),
            num_frames=config['data_dims'][1],  # T
            tubelet_size=config.get('tubelet_size', 2),
            in_chans=config['data_dims'][0],  # C
            rep_dim=config['rep_dim']
        ).to(device)
    elif config['vit_size'] == 'base':
        online_encoder = vit_base_video(
            img_size=config['data_dims'][2],
            patch_size=config.get('patch_size', 8),
            num_frames=config['data_dims'][1],
            tubelet_size=config.get('tubelet_size', 2),
            in_chans=config['data_dims'][0],
            rep_dim=config['rep_dim']
        ).to(device)
    else:
        raise ValueError(f"Unknown ViT size: {config.get('vit_size')}")
    
    # Initialize target encoder (momentum)
    target_encoder = deepcopy(online_encoder).to(device)
    for param in target_encoder.parameters():
        param.requires_grad = False
    
    # Initialize JEPA model
    model = JEPA(online_encoder, config['rep_dim']).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['jepa_lr'],
        weight_decay=config.get('weight_decay', 0.05),
        betas=(0.9, 0.95)
    )
    
    # Momentum coefficient for EMA
    MOMENTUM = config.get('momentum', 0.996)
    
    # Training loop
    for epoch in range(config['jepa_epochs']):
        epoch_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            X = batch['X'].to(device)
            
            # Apply tube masking
            X_context, X_target, masks = apply_tube_masking(
                X,
                mask_ratio=config.get('mask_ratio', 0.75),
                num_masks=config.get('num_mask_blocks', 4)
            )
            
            # Compute target representations (stop-gradient, momentum encoder)
            with torch.no_grad():
                target_reps = target_encoder(X_target).detach()
            
            # Compute predicted representations (online encoder + predictor)
            optimizer.zero_grad()
            predicted_reps = model(X_context, compute_target=False)
            
            # JEPA loss (L1 distance in representation space)
            loss = nn.functional.l1_loss(predicted_reps, target_reps)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Momentum update of target encoder (EMA)
            with torch.no_grad():
                for online_param, target_param in zip(online_encoder.parameters(), target_encoder.parameters()):
                    target_param.data.mul_(MOMENTUM).add_(online_param.data, alpha=1.0 - MOMENTUM)
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{config['jepa_epochs']}, Loss: {avg_loss:.4f}")
    
    # Return the trained momentum encoder
    return target_encoder.eval()
