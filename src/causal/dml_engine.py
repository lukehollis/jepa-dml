import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from src.models.jepa import train_jepa
from src.causal.nuisance_models import train_nuisance_models

def generate_representations(encoder, data_loader, device):
    """Helper to extract R, T, Y using a trained encoder efficiently."""
    encoder.eval()
    R_list, T_list, Y_list = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            X, T, Y = batch['X'].to(device), batch['T'].to(device), batch['Y'].to(device)
            R = encoder(X)
            R_list.append(R)
            T_list.append(T)
            Y_list.append(Y)
    # Return as tensors already on the device
    return torch.cat(R_list), torch.cat(T_list), torch.cat(Y_list)


def execute_dml_workflow(dataset, config, device, train_encoder_func=train_jepa):
    """
    Implements the full K-Fold Cross-Fitting DML procedure (ICCV Algorithm 1).
    """
    kf = KFold(n_splits=config['k_folds'], shuffle=True, random_state=42)
    scores = []

    # Iterate over folds
    for fold, (train_index, test_index) in enumerate(kf.split(range(len(dataset)))):
        print(f"\n--- Processing Fold {fold+1}/{config['k_folds']} ---")

        # Create DataLoaders for this fold
        train_subset = Subset(dataset, train_index)
        test_subset = Subset(dataset, test_index)
        
        # Loaders for Representation Learning (needs shuffle)
        train_loader_repr_learn = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
        
        # Loaders for representation generation (no shuffle needed)
        train_loader_repr = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=False)
        test_loader_repr = DataLoader(test_subset, batch_size=config['batch_size'], shuffle=False)

        # === Stage 1: Fold-Specific Representation Learning ===
        # CRITICAL: Train Encoder from scratch only on the training data of this fold.
        encoder = train_encoder_func(train_loader_repr_learn, config, device)

        # Generate representations R using the fold-specific encoder
        print("  Generating representations (R)...")
        R_train, T_train, Y_train = generate_representations(encoder, train_loader_repr, device)
        R_test, T_test, Y_test = generate_representations(encoder, test_loader_repr, device)

        # === Stage 2: Nuisance Model Training ===
        print("  Training Nuisance Models (f(R), µ, π)...")
        causal_nets, propensity_model = train_nuisance_models(
            R_train, Y_train, T_train, config, device
        )

        # === Stage 3: Out-of-Sample Prediction and Score Calculation ===
        print("  Calculating DML Scores...")

        with torch.no_grad():
            # Get f(R_test)
            f_R_test = causal_nets.get_proxy(R_test)
            
            # Predict Outcomes (µ0, µ1)
            mu0_test, mu1_test = causal_nets.predict_outcomes(f_R_test)
            mu0_test = mu0_test.squeeze()
            mu1_test = mu1_test.squeeze()

            # Predict Propensity Scores (π)
            pi_test = propensity_model(f_R_test).squeeze()
            
            # Phase 4 Diagnostic: Check overlap and clip for stability
            pi_test = torch.clamp(pi_test, 0.01, 0.99)

            # Calculate Neyman-Orthogonal Scores (ψ) - AIPW Estimator
            # ψ = (T(Y-µ1)/π) - ((1-T)(Y-µ0)/(1-π)) + µ1 - µ0
            term1 = (T_test * (Y_test - mu1_test)) / pi_test
            term2 = ((1 - T_test) * (Y_test - mu0_test)) / (1 - pi_test)
            term3 = mu1_test - mu0_test

            fold_scores = (term1 - term2 + term3).cpu().numpy()
            scores.extend(fold_scores)

    # === Final ATE Estimation ===
    ate_estimate = np.mean(scores)
    variance = np.var(scores) / len(scores)
    std_err = np.sqrt(variance)

    return ate_estimate, std_err
