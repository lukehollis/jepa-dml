"""Training script for JEPA-DML Causal Inference Engine"""
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import SpatiotemporalCausalDataset
from src.causal.dml_engine import execute_jepa_dml_workflow

# Configuration
# NOTE: These parameters are reduced for demo. Real-world application requires more training.
CONFIG = {
    'n_samples': 200,              # Reduced sample size for demo
    'data_dims': (3, 8, 32, 32),   # C, T, H, W (Reduced resolution)
    'rep_dim': 128,                # Dimension of JEPA representation R
    'proxy_dim': 32,               # Dimension of confounder proxy f(R)
    'k_folds': 3,                  # Number of cross-fitting folds
    'batch_size': 16,
    
    # JEPA training params
    'vit_size': 'small',           # 'small' or 'base' (ViT-S or ViT-B)
    'patch_size': 8,               # Patch size for ViT
    'tubelet_size': 2,             # Temporal grouping size
    'jepa_epochs': 20,             # Epochs for JEPA training (per fold)
    'jepa_lr': 1e-4,               # Learning rate for JEPA
    'weight_decay': 0.05,          # Weight decay for AdamW
    'momentum': 0.996,             # EMA momentum for target encoder
    'mask_ratio': 0.75,            # Fraction of patches to mask
    'num_mask_blocks': 4,          # Number of tube mask blocks
    
    # DML training params
    'dml_epochs': 10,              # Epochs for nuisance models (per fold)
}

def main():
    print("Initializing JEPA-DML Causal Inference Engine Simulation...")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 1. Generate/Load Data (Phase 1)
    dataset = SpatiotemporalCausalDataset(CONFIG['n_samples'], CONFIG['data_dims'])
    print(f"Dataset simulated with {len(dataset)} samples. True ATE = 1.0.")

    # 2. & 3. Execute JEPA-DML Workflow (Phases 2 & 3)
    print("\nStarting JEPA-DML Workflow...")
    print("NOTE: This involves retraining the JEPA encoder K times, once within each fold.")

    # Execute the workflow
    estimated_ate, std_err = execute_jepa_dml_workflow(dataset, CONFIG, DEVICE)

    # 4. Evaluation (Phase 4)
    print("\n--- Results ---")
    print(f"Estimated ATE: {estimated_ate:.4f}")
    print(f"Standard Error: {std_err:.4f}")
    print(f"95% CI: [{estimated_ate - 1.96*std_err:.4f}, {estimated_ate + 1.96*std_err:.4f}]")
    print(f"True ATE: 1.0")

if __name__ == '__main__':
    main()
