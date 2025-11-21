# JEPA-DML Causal Inference Engine

A global causal inference engine using Spatiotemporal Joint Embedding Predictive Architecture (JEPA) world models and Double Machine Learning (DML) for robust causal analysis from high-dimensional data.

## Overview

This implementation combines:
- **V-JEPA-inspired self-supervised learning** for spatiotemporal representation learning
- **Double Machine Learning (DML)** with K-fold cross-fitting for causal effect estimation
- **Confounder proxy networks** to handle latent confounding in observational data

## Setup

### Using Conda (Recommended)

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate causal_engine
```

### Using pip

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the simulation to verify the installation:

```bash
python main.py
```

This will:
1. Generate simulated spatiotemporal data with latent confounders
2. Train the JEPA world model using K-fold cross-fitting
3. Estimate the Average Treatment Effect (ATE) using DML
4. Output results with confidence intervals

Expected output should show an estimated ATE close to the true value of 1.0.

## Project Structure

```
causal_engine/
├── src/
│   ├── data/
│   │   └── dataset.py           # Spatiotemporal data simulation
│   ├── models/
│   │   ├── encoder.py           # Spatiotemporal encoder (3D CNN)
│   │   └── jepa.py              # JEPA architecture and training
│   └── causal/
│       ├── nuisance_models.py   # Proxy, outcome, and propensity models
│       └── dml_engine.py        # DML cross-fitting workflow
├── main.py                      # Main execution script
├── environment.yml              # Conda environment specification
└── requirements.txt             # Pip requirements
```

## Configuration

Edit the `CONFIG` dictionary in `main.py` to adjust:
- `n_samples`: Number of data samples
- `data_dims`: Spatiotemporal dimensions (C, T, H, W)
- `rep_dim`: JEPA representation dimension
- `k_folds`: Number of cross-fitting folds
- `jepa_epochs`: JEPA training epochs per fold
- `dml_epochs`: Nuisance model training epochs

## Key Features

### Rigorous DML Implementation
- Per-fold JEPA training from scratch to prevent data leakage
- Out-of-sample predictions for valid causal inference
- Neyman-orthogonal score functions (AIPW estimator)

### Spatiotemporal World Model
- Self-supervised learning via masked prediction
- Momentum encoder for stable target representations
- Captures temporal dependencies in video/time-series data

### Confounder Proxy Network
- TarNet-style architecture for outcome prediction
- Separate propensity model for treatment probability
- Handles high-dimensional latent confounding

## Citation

Based on research from:
- V-JEPA 2 (Meta FAIR)
- Self-Supervised Predictive Representations for Causal Inference
- GenAI-Powered Inference (GPI) framework

## License

MIT
