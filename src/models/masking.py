"""
Spatiotemporal tube masking for JEPA.
Implements structured masking strategies for video data.
"""
import torch
import numpy as np


def generate_tube_mask(
    batch_size,
    num_frames,
    num_patches_per_frame,
    mask_ratio=0.75,
    tubelet_size=2,
    aspect_ratio=(0.75, 1.5),
    min_area=0.1,
    max_area=0.3,
    num_masks=4
):
    """
    Generate spatiotemporal tube masks for video.
    
    Args:
        batch_size: Number of samples in batch
        num_frames: Number of temporal frames
        num_patches_per_frame: Number of spatial patches per frame (H_patches * W_patches)
        mask_ratio: Target ratio of masked patches
        tubelet_size: Temporal grouping size
        aspect_ratio: (min, max) aspect ratio for mask blocks
        min_area: Minimum area of mask block as fraction
        max_area: Maximum area of mask block as fraction
        num_masks: Number of mask blocks to generate
    
    Returns:
        mask: (batch_size, total_patches) boolean mask tensor
    """
    device = 'cpu'  # Will be moved to correct device later
    
    # Calculate dimensions
    H_patches = W_patches = int(np.sqrt(num_patches_per_frame))
    T_tubelets = num_frames // tubelet_size
    total_patches = T_tubelets * num_patches_per_frame
    
    masks = []
    
    for _ in range(batch_size):
        mask = torch.zeros(T_tubelets, H_patches, W_patches, dtype=torch.bool)
        
        # Generate multiple mask blocks
        for _ in range(num_masks):
            # Random aspect ratio
            aspect = np.random.uniform(*aspect_ratio)
            
            # Random area
            area = np.random.uniform(min_area, max_area)
            
            # Calculate height and width
            h = int(np.sqrt(area * num_patches_per_frame / aspect))
            w = int(aspect * h)
            h = max(1, min(h, H_patches))
            w = max(1, min(w, W_patches))
            
            # Random temporal extent (tube length)
            t_len = np.random.randint(1, T_tubelets + 1)
            
            # Random position
            t_start = np.random.randint(0, T_tubelets - t_len + 1)
            y_start = np.random.randint(0, H_patches - h + 1)
            x_start = np.random.randint(0, W_patches - w + 1)
            
            # Apply mask
            mask[t_start:t_start+t_len, y_start:y_start+h, x_start:x_start+w] = True
        
        # Flatten to (total_patches,)
        mask_flat = mask.reshape(-1)
        masks.append(mask_flat)
    
    masks = torch.stack(masks, dim=0)  # (batch_size, total_patches)
    
    # Ensure we hit target mask ratio
    current_ratio = masks.float().mean()
    if current_ratio < mask_ratio:
        # Randomly mask additional patches
        num_additional = int((mask_ratio - current_ratio) * total_patches * batch_size)
        for i in range(batch_size):
            available = (~masks[i]).nonzero(as_tuple=True)[0]
            if len(available) > 0:
                n_sample = min(num_additional // batch_size, len(available))
                if n_sample > 0:
                    indices = available[torch.randperm(len(available))[:n_sample]]
                    masks[i, indices] = True
    
    return masks


def apply_mask(x, mask):
    """
    Apply mask to patch embeddings.
    
    Args:
        x: (B, N, D) patch embeddings
        mask: (B, N) boolean mask (True = keep, False = mask)
    
    Returns:
        x_masked: (B, N_kept, D) masked embeddings
        mask_indices: (B, N_kept) indices of kept patches
    """
    B, N, D = x.shape
    x_masked = []
    mask_indices = []
    
    for i in range(B):
        keep_indices = mask[i].nonzero(as_tuple=True)[0]
        x_masked.append(x[i, keep_indices])
        mask_indices.append(keep_indices)
    
    # Pad to same length for batching
    max_len = max(len(m) for m in x_masked)
    x_padded = torch.zeros(B, max_len, D, device=x.device, dtype=x.dtype)
    
    for i in range(B):
        x_padded[i, :len(x_masked[i])] = x_masked[i]
    
    return x_padded, mask_indices


def random_tube_masking(
    batch_size,
    num_frames,
    num_patches_per_frame,
    tubelet_size=2,
    mask_ratio=0.75
):
    """
    Simple random tube masking strategy.
    Randomly masks complete tubes (temporal sequences at spatial locations).
    
    Args:
        batch_size: Batch size
        num_frames: Number of frames
        num_patches_per_frame: Number of patches per frame
        tubelet_size: Temporal grouping
        mask_ratio: Fraction to mask
    
    Returns:
        mask: (batch_size, total_patches) boolean mask
    """
    T_tubelets = num_frames // tubelet_size
    total_patches = T_tubelets * num_patches_per_frame
    
    masks = []
    for _ in range(batch_size):
        # Randomly decide which tubes to mask
        tube_mask = torch.rand(num_patches_per_frame) < mask_ratio
        
        # Expand to all time steps
        mask = tube_mask.unsqueeze(0).expand(T_tubelets, -1)
        mask = mask.reshape(-1)
        
        masks.append(mask)
    
    return torch.stack(masks, dim=0)


def future_frame_masking(
    batch_size,
    num_frames,
    num_patches_per_frame,
    tubelet_size=2,
    num_context_frames=None
):
    """
    Mask all future frames, keeping only context frames.
    This is useful for causal prediction tasks.
    
    Args:
        batch_size: Batch size
        num_frames: Total number of frames
        num_patches_per_frame: Patches per frame
        tubelet_size: Temporal grouping
        num_context_frames: Number of initial frames to keep (default: half)
    
    Returns:
        context_mask: (batch_size, total_patches) boolean for context
        target_mask: (batch_size, total_patches) boolean for targets
    """
    T_tubelets = num_frames // tubelet_size
    
    if num_context_frames is None:
        num_context_frames = T_tubelets // 2
    else:
        num_context_frames = num_context_frames // tubelet_size
    
    total_patches = T_tubelets * num_patches_per_frame
    
    context_masks = []
    target_masks = []
    
    for _ in range(batch_size):
        context_mask = torch.zeros(T_tubelets, num_patches_per_frame, dtype=torch.bool)
        target_mask = torch.zeros(T_tubelets, num_patches_per_frame, dtype=torch.bool)
        
        # Keep context frames
        context_mask[:num_context_frames] = True
        
        # Target is all frames
        target_mask[:] = True
        
        context_masks.append(context_mask.reshape(-1))
        target_masks.append(target_mask.reshape(-1))
    
    return torch.stack(context_masks), torch.stack(target_masks)
