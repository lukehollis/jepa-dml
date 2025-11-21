import numpy as np
import pandas as pd
import torch

def calculate_ate_error(ate_est, ate_true=1.0):
    """
    Calculate absolute error in ATE estimation.
    """
    return np.abs(ate_est - ate_true)

def calculate_pehe(ite_est, ite_true=1.0):
    """
    Calculate Precision in Estimation of Heterogeneous Effect (PEHE).
    RMSE between predicted ITE and true ITE.
    """
    # If ite_true is a scalar, broadcast it
    if np.isscalar(ite_true):
        ite_true = np.full_like(ite_est, ite_true)
        
    return np.sqrt(np.mean((ite_true - ite_est)**2))

class BaselineEvaluator:
    """
    Comparison runner for causal baselines.
    """
    def __init__(self, results_dir='results/baselines'):
        self.results_dir = results_dir
        self.results = []

    def log_result(self, model_name, ate_est, ate_error, pehe=None, training_time=None):
        result = {
            'model': model_name,
            'ate_est': ate_est,
            'ate_error': ate_error,
            'pehe': pehe,
            'training_time': training_time
        }
        self.results.append(result)
        print(f"[{model_name}] ATE Est: {ate_est:.4f}, Error: {ate_error:.4f}")

    def save_results(self, filename='comparison_results.csv'):
        df = pd.DataFrame(self.results)
        import os
        os.makedirs(self.results_dir, exist_ok=True)
        df.to_csv(os.path.join(self.results_dir, filename), index=False)
        print(f"Results saved to {os.path.join(self.results_dir, filename)}")
        return df
