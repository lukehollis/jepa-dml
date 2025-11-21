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
