import torch
from src.data.dataset import SpatiotemporalCausalDataset
from torch.utils.data import DataLoader
from app.pipeline.orchestrator import CausalOrchestrator
import time

def main():
    print("--- Testing Causal Pipeline (Orchestrator) ---")
    
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    orchestrator = CausalOrchestrator(device=device)
    
    config = {
        'data_dims': (3, 16, 64, 64),
        'rep_dim': 128,
        'proxy_dim': 64,
        'batch_size': 16,
        'k_folds': 2,
        'jepa_epochs': 2, # Very short for testing
        'dml_epochs': 2,
        'vit_size': 'small'
    }
    
    # 2. Create Data
    print("\n1. Creating Dataset...")
    dataset = SpatiotemporalCausalDataset(n_samples=50)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 3. Run 1: Should Train Model
    print("\n2. Run 1 (Expected: Training)...")
    start = time.time()
    result1 = orchestrator.run_analysis(loader, config)
    dur1 = time.time() - start
    print(f"Result 1: ATE={result1['ate']:.4f} (Duration: {dur1:.2f}s)")
    
    # 4. Run 2: Should Load Cache
    print("\n3. Run 2 (Expected: Cache Hit)...")
    start = time.time()
    result2 = orchestrator.run_analysis(loader, config)
    dur2 = time.time() - start
    print(f"Result 2: ATE={result2['ate']:.4f} (Duration: {dur2:.2f}s)")
    
    if dur2 < dur1:
        print("\n[SUCCESS] Second run was faster (Cache Hit).")
    else:
        print("\n[WARNING] Second run was not significantly faster.")
        
    print("\nPipeline test complete.")

if __name__ == "__main__":
    main()
