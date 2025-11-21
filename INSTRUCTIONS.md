This response provides a comprehensive Python blueprint for implementing the task plan: training a Spatiotemporal JEPA world model and integrating it with Double Machine Learning (DML) for causal inference.

This implementation adheres strictly to the rigorous methodology outlined in the provided research papers (GPI and the ICCV submission), emphasizing the necessity of retraining the representation model (JEPA) within each fold of the DML cross-fitting procedure to prevent data leakage and ensure valid causal inference.

### Project Structure

We adopt a modular structure for maintainability and clarity.

```
jepa_dml_project/
├── src/
│   ├── data/
│   │   └── dataset.py           # Data handling and simulation
│   ├── models/
│   │   ├── encoder.py           # Spatiotemporal Encoder (ViT/CNN Placeholder)
│   │   └── jepa.py              # JEPA wrapper and training loop
│   ├── causal/
│   │   ├── nuisance_models.py   # Proxy f(R), Outcome (µ), Propensity (π)
│   │   └── dml_engine.py        # DML Cross-fitting Logic
└── main.py                      # Main execution script
```

### 1\. `src/data/dataset.py` (Phase 1: Data Handling)

A dataset class to simulate spatiotemporal data (X) with associated treatments (T) and outcomes (Y), influenced by latent confounders (U).

```python
import torch
from torch.utils.data import Dataset
import numpy as np

class SpatiotemporalCausalDataset(Dataset):
    """
    Simulates a dataset where latent confounders (U) influence X, T, and Y.
    """
    def __init__(self, n_samples=1000, data_dims=(3, 16, 64, 64)): # C, T, H, W
        self.n_samples = n_samples
        self.data_dims = data_dims
        self.data = self._generate_simulated_data()

    def _generate_simulated_data(self):
        # 1. Generate Latent Confounders (U)
        U = np.random.normal(0, 1, size=(self.n_samples, 5))

        # 2. Generate Spatiotemporal Data (X) influenced by U
        X = np.random.rand(self.n_samples, *self.data_dims).astype(np.float32)
        # Inject confounding signal (e.g., U[:, 0] affects brightness)
        X += U[:, 0].reshape(-1, 1, 1, 1, 1) * 0.5

        # 3. Generate Treatment (T) influenced by U
        wT = np.array([0.5, -0.5, 0.2, 0.1, -0.1])
        prob_T = 1 / (1 + np.exp(-(U @ wT + np.random.normal(0, 0.5, self.n_samples))))
        T = (prob_T > 0.5).astype(np.float32)

        # 4. Generate Outcome (Y) (True ATE = 1.0)
        wY = np.array([0.3, 0.3, 0.1, 0.1, 0.1])
        # Non-linear confounding influence (U**2)
        Y = 1.0 * T + (U**2) @ wY + np.random.normal(0, 1, self.n_samples)
        Y = Y.astype(np.float32)

        return {'X': X, 'T': T, 'Y': Y}

    def get_data_arrays(self):
        # Helper for accessing raw arrays if needed, though we will use DataLoaders
        return self.data['X'], self.data['T'], self.data['Y']

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'X': torch.tensor(self.data['X'][idx]),
            'T': torch.tensor(self.data['T'][idx]),
            'Y': torch.tensor(self.data['Y'][idx]),
        }
```

### 2\. `src/models/encoder.py` (Phase 2: Spatiotemporal Encoder)

A placeholder for the encoder architecture. A real implementation requires a Spatiotemporal Transformer (e.g., ViViT).

```python
import torch.nn as nn

class SpatiotemporalEncoder(nn.Module):
    """
    Placeholder for a Spatiotemporal Transformer.
    We use a simplified 3D CNN here for demonstration purposes to handle 5D input.
    """
    def __init__(self, input_dims, rep_dim):
        super().__init__()
        C, T, H, W = input_dims
        
        self.encoder = nn.Sequential(
            # Input: (B, C, T, H, W)
            nn.Conv3d(C, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)), # Global pooling
            nn.Flatten(),
            nn.Linear(64, rep_dim)
        )

    def forward(self, x):
        if x.dim() != 5:
             raise ValueError(f"Expected 5D tensor (B, C, T, H, W), got {x.dim()}D")
        return self.encoder(x)
```

### 3\. `src/models/jepa.py` (Phase 2: JEPA Framework and Training)

The core JEPA architecture and the self-supervised training loop.

```python
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
```

### 4\. `src/causal/nuisance_models.py` (Phase 3: DML Components)

Defines the TarNet-inspired architecture for the Confounder Proxy (f(R)) and Outcome Model (µ), and the Propensity Model (π).

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CausalNetworks(nn.Module):
    """
    Encapsulates the Confounder Proxy f(R) and the Outcome Model µt(f(R)).
    Trained jointly (TarNet style).
    """
    def __init__(self, rep_dim, proxy_dim):
        super().__init__()
        # Confounder Proxy Network (f(R))
        self.proxy_net = nn.Sequential(
            nn.Linear(rep_dim, 128),
            nn.ReLU(),
            nn.Linear(128, proxy_dim),
            nn.LayerNorm(proxy_dim)
        )
        # Outcome Model Heads (µ0 and µ1)
        self.head_0 = nn.Sequential(nn.Linear(proxy_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_1 = nn.Sequential(nn.Linear(proxy_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def get_proxy(self, R):
        return self.proxy_net(R)

    def forward(self, R, T):
        # Used during training
        f_R = self.get_proxy(R)
        out_0 = self.head_0(f_R)
        out_1 = self.head_1(f_R)
        # Select output based on actual treatment T
        return torch.where(T.view(-1, 1) == 0, out_0, out_1)

    def predict_outcomes(self, f_R):
        # Used during inference/score calculation
        return self.head_0(f_R), self.head_1(f_R)

class PropensityModel(nn.Module):
    """
    The Treatment Model π(f(R)). Input must be f(R), not R.
    """
    def __init__(self, proxy_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(proxy_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Ensure output is a probability
        )

    def forward(self, f_R):
        return self.model(f_R)

def train_nuisance_models(R_train, Y_train, T_train, config, device):
    """
    Trains the Proxy/Outcome networks jointly, and the Propensity network separately.
    """
    rep_dim = R_train.shape[1]
    proxy_dim = config['proxy_dim']

    # Initialize models
    causal_nets = CausalNetworks(rep_dim, proxy_dim).to(device)
    propensity_model = PropensityModel(proxy_dim).to(device)

    # Setup optimizers and criteria
    opt_causal = optim.Adam(causal_nets.parameters(), lr=1e-3)
    opt_prop = optim.Adam(propensity_model.parameters(), lr=1e-3)
    crit_outcome = nn.MSELoss()
    crit_prop = nn.BCELoss()

    # Prepare data loaders (Assumes R_train fits in memory)
    dataset = TensorDataset(R_train, Y_train, T_train)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    for epoch in range(config['dml_epochs']):
        for R_batch, Y_batch, T_batch in dataloader:
            
            # 1. Train Causal Nets (Proxy f(R) + Outcome µ)
            opt_causal.zero_grad()
            Y_pred = causal_nets(R_batch, T_batch)
            loss_outcome = crit_outcome(Y_pred.squeeze(), Y_batch)
            loss_outcome.backward()
            opt_causal.step()

            # 2. Train Propensity Model (π)
            opt_prop.zero_grad()
            # Crucial: Get f(R) and detach it. We train π on the proxy learned by the outcome model.
            with torch.no_grad():
                f_R_batch = causal_nets.get_proxy(R_batch).detach()
            
            T_pred_prob = propensity_model(f_R_batch)
            loss_prop = crit_prop(T_pred_prob.squeeze(), T_batch)
            loss_prop.backward()
            opt_prop.step()

    return causal_nets.eval(), propensity_model.eval()
```

### 5\. `src/causal/dml_engine.py` (Phase 3: DML Cross-Fitting)

The core engine implementing the rigorous K-Fold procedure, ensuring that the JEPA representation is learned independently within each fold.

```python
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


def execute_jepa_dml_workflow(dataset, config, device):
    """
    Implements the full K-Fold Cross-Fitting DML procedure (ICCV Algorithm 1).
    """
    kf = KFold(n_splits=config['k_folds'], shuffle=True, random_state=42)
    scores = []
    input_dims = dataset.data_dims

    # Iterate over folds
    for fold, (train_index, test_index) in enumerate(kf.split(range(len(dataset)))):
        print(f"\n--- Processing Fold {fold+1}/{config['k_folds']} ---")

        # Create DataLoaders for this fold
        train_subset = Subset(dataset, train_index)
        test_subset = Subset(dataset, test_index)
        
        # Loaders for JEPA training (needs shuffle)
        train_loader_jepa = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
        
        # Loaders for representation generation (no shuffle needed)
        train_loader_repr = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=False)
        test_loader_repr = DataLoader(test_subset, batch_size=config['batch_size'], shuffle=False)

        # === Stage 1: Fold-Specific Representation Learning ===
        # CRITICAL: Train JEPA from scratch only on the training data of this fold.
        jepa_encoder = train_jepa(train_loader_jepa, input_dims, config, device)

        # Generate representations R using the fold-specific encoder
        print("  Generating representations (R)...")
        R_train, T_train, Y_train = generate_representations(jepa_encoder, train_loader_repr, device)
        R_test, T_test, Y_test = generate_representations(jepa_encoder, test_loader_repr, device)

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

```

### 6\. `main.py` (Execution Script)

The main entry point to initialize the configuration and run the pipeline.

```python
import torch
# Import modules assuming the structure defined above
from src.data.dataset import SpatiotemporalCausalDataset
from src.causal.dml_engine import execute_jepa_dml_workflow

# Configuration
# NOTE: These parameters are significantly reduced for demonstration. 
# Real-world application requires extensive training and larger models.
CONFIG = {
    'n_samples': 200,           # Reduced sample size
    'data_dims': (3, 8, 32, 32), # C, T, H, W (Reduced resolution)
    'rep_dim': 128,             # Dimension of JEPA representation R
    'proxy_dim': 32,            # Dimension of confounder proxy f(R)
    'k_folds': 3,               # Reduced folds
    'batch_size': 16,
    'jepa_epochs': 2,           # Epochs for JEPA training (per fold)
    'jepa_lr': 1e-4,
    'dml_epochs': 5,            # Epochs for nuisance models (per fold)
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
    # estimated_ate, std_err = execute_jepa_dml_workflow(dataset, CONFIG, DEVICE)

    # 4. Evaluation (Phase 4)
    print("\n--- Workflow Structure Complete ---")
    # if 'estimated_ate' in locals():
    #     print(f"Estimated ATE: {estimated_ate:.4f}")
    #     print(f"Standard Error: {std_err:.4f}")
    #     print(f"95% CI: [{estimated_ate - 1.96*std_err:.4f}, {estimated_ate + 1.96*std_err:.4f}]")
    
    print("\nMain execution loop (`execute_jepa_dml_workflow`) is commented out.")
    print("Uncomment to run the simulation. Ensure a suitable GPU environment is available.")

if __name__ == '__main__':
    # To run this, ensure the files are saved in the defined structure.
    main()
```
