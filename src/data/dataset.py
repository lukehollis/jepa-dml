import torch
from torch.utils.data import Dataset
import numpy as np

class SpatiotemporalCausalDataset(Dataset):
    """
    Simulates a dataset where latent confounders (U) influence X, T, and Y.
    """
    def __init__(self, n_samples=1000, data_dims=(3, 16, 64, 64), return_confounders=False): # C, T, H, W
        self.n_samples = n_samples
        self.data_dims = data_dims
        self.return_confounders = return_confounders
        self.data = self._generate_simulated_data()

    def _generate_simulated_data(self):
        # 1. Generate Latent Confounders (U)
        U = np.random.normal(0, 1, size=(self.n_samples, 5)).astype(np.float32)

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

        return {'X': X, 'T': T, 'Y': Y, 'U': U}

    def get_data_arrays(self):
        # Helper for accessing raw arrays if needed, though we will use DataLoaders
        return self.data['X'], self.data['T'], self.data['Y']

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        item = {
            'X': torch.tensor(self.data['X'][idx]),
            'T': torch.tensor(self.data['T'][idx]),
            'Y': torch.tensor(self.data['Y'][idx]),
        }
        if self.return_confounders:
            item['U'] = torch.tensor(self.data['U'][idx])
        return item
