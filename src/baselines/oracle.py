import torch
import torch.nn as nn
import torch.optim as optim

class OracleModel(nn.Module):
    """
    Oracle baseline that has access to latent confounders (U).
    Predicts Y from U and T.
    """
    def __init__(self, u_dim=5):
        super().__init__()
        # Input: U (u_dim) + T (1)
        self.model = nn.Sequential(
            nn.Linear(u_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, u, t):
        # Concatenate U and T
        x = torch.cat([u, t.view(-1, 1)], dim=1)
        return self.model(x)

def train_oracle(train_loader, config, device):
    """
    Train the Oracle model.
    Requires 'U' to be present in the batch.
    """
    # Infer u_dim from the first batch
    first_batch = next(iter(train_loader))
    if 'U' not in first_batch:
        raise ValueError("Oracle training requires 'U' (confounders) in the dataset.")
    
    u_dim = first_batch['U'].shape[1]
    
    model = OracleModel(u_dim=u_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"Starting Oracle training for {config['epochs']} epochs...")
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            u = batch['U'].to(device)
            t = batch['T'].to(device)
            y = batch['Y'].to(device)
            
            optimizer.zero_grad()
            y_pred = model(u, t)
            loss = criterion(y_pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}, Loss: {epoch_loss/len(train_loader):.4f}")
            
    return model

def predict_oracle(model, loader, device):
    """
    Calculate ATE using the Oracle model.
    ATE = E[Y(1) - Y(0)]
    """
    model.eval()
    y1_preds = []
    y0_preds = []
    
    with torch.no_grad():
        for batch in loader:
            u = batch['U'].to(device)
            
            # Create counterfactual treatments
            t1 = torch.ones(u.size(0), device=device)
            t0 = torch.zeros(u.size(0), device=device)
            
            y1 = model(u, t1)
            y0 = model(u, t0)
            
            y1_preds.append(y1.cpu())
            y0_preds.append(y0.cpu())
            
    y1_all = torch.cat(y1_preds)
    y0_all = torch.cat(y0_preds)
    
    ate = (y1_all - y0_all).mean().item()
    return ate
