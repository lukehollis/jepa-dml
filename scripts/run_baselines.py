import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from src.data.dataset import SpatiotemporalCausalDataset
from src.causal.dml_engine import execute_dml_workflow
from src.models.jepa import train_jepa
from src.baselines.vicreg import train_vicreg
from src.baselines.dragonnet import train_dragonnet, predict_dragonnet
from src.baselines.oracle import train_oracle, predict_oracle
from src.evaluation.comparison import BaselineEvaluator, calculate_ate_error, calculate_pehe
import time

def evaluate_dragonnet_cv(dataset, config, device):
    kf = KFold(n_splits=config['k_folds'], shuffle=True, random_state=42)
    ate_estimates = []
    
    print(f"\n--- Running DragonNet ({config['k_folds']}-Fold CV) ---")
    
    for fold, (train_index, test_index) in enumerate(kf.split(range(len(dataset)))):
        print(f"  Fold {fold+1}/{config['k_folds']}")
        train_subset = Subset(dataset, train_index)
        test_subset = Subset(dataset, test_index)
        
        train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=config['batch_size'], shuffle=False)
        
        # Train
        model = train_dragonnet(train_loader, test_loader, config, device)
        
        # Predict
        ate = predict_dragonnet(model, test_loader, device)
        ate_estimates.append(ate)
        
    return np.mean(ate_estimates), np.std(ate_estimates) / np.sqrt(len(ate_estimates))

def evaluate_oracle_cv(dataset, config, device):
    kf = KFold(n_splits=config['k_folds'], shuffle=True, random_state=42)
    ate_estimates = []
    
    print(f"\n--- Running Oracle ({config['k_folds']}-Fold CV) ---")
    
    for fold, (train_index, test_index) in enumerate(kf.split(range(len(dataset)))):
        print(f"  Fold {fold+1}/{config['k_folds']}")
        train_subset = Subset(dataset, train_index)
        test_subset = Subset(dataset, test_index)
        
        train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=config['batch_size'], shuffle=False)
        
        # Train
        model = train_oracle(train_loader, config, device)
        
        # Predict
        ate = predict_oracle(model, test_loader, device)
        ate_estimates.append(ate)
        
    return np.mean(ate_estimates), np.std(ate_estimates) / np.sqrt(len(ate_estimates))

def main():
    # Configuration
    config = {
        'data_dims': (3, 16, 64, 64), # C, T, H, W
        'rep_dim': 256,
        'proxy_dim': 64,
        'batch_size': 32,
        'k_folds': 5,
        'jepa_epochs': 5,    # Short for testing
        'dml_epochs': 5,     # Short for testing
        'epochs': 5,         # For DragonNet/Oracle
        'jepa_lr': 1e-4,
        'vicreg_lr': 1e-4,
        'mask_ratio': 0.75,
        'vit_size': 'small',
        'dragonnet_alpha': 1.0,
        'dragonnet_beta': 1.0
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Evaluator
    evaluator = BaselineEvaluator()
    
    # Load Data
    print("Generating Spatiotemporal Dataset...")
    dataset = SpatiotemporalCausalDataset(n_samples=200, return_confounders=True) # Small sample for quick test
    
    # 1. Oracle Baseline
    start_time = time.time()
    ate_oracle, err_oracle = evaluate_oracle_cv(dataset, config, device)
    evaluator.log_result('Oracle', ate_oracle, calculate_ate_error(ate_oracle), training_time=time.time()-start_time)
    
    # 2. DragonNet Baseline
    start_time = time.time()
    ate_dragon, err_dragon = evaluate_dragonnet_cv(dataset, config, device)
    evaluator.log_result('DragonNet', ate_dragon, calculate_ate_error(ate_dragon), training_time=time.time()-start_time)
    
    # 3. VICReg-DML
    start_time = time.time()
    print("\n--- Running VICReg-DML ---")
    ate_vicreg, err_vicreg = execute_dml_workflow(dataset, config, device, train_encoder_func=train_vicreg)
    evaluator.log_result('VICReg-DML', ate_vicreg, calculate_ate_error(ate_vicreg), training_time=time.time()-start_time)
    
    # 4. JEPA-DML (Our Method)
    start_time = time.time()
    print("\n--- Running JEPA-DML (Proposed) ---")
    ate_jepa, err_jepa = execute_dml_workflow(dataset, config, device, train_encoder_func=train_jepa)
    evaluator.log_result('JEPA-DML', ate_jepa, calculate_ate_error(ate_jepa), training_time=time.time()-start_time)
    
    # Save Results
    evaluator.save_results()
    print("\nBaseline comparison completed successfully.")

if __name__ == "__main__":
    main()
