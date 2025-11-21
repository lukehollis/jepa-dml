"""
Download and prepare standard causal inference benchmark datasets.
"""
import urllib.request
import os
from pathlib import Path


def download_ihdp():
    """Download IHDP (Infant Health and Development Program) dataset."""
    print("Downloading IHDP dataset...")
    
    base_url = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/"
    datasets_dir = Path("datasets/ihdp")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # Download 100 replications
    for i in range(1, 101):
        filename = f"ihdp_npci_{i}.csv"
        url = base_url + filename
        output_path = datasets_dir / filename
        
        if not output_path.exists():
            try:
                urllib.request.urlretrieve(url, output_path)
                if i % 10 == 0:
                    print(f"  Downloaded {i}/100 replications")
            except Exception as e:
                print(f"  Failed to download {filename}: {e}")
                break
    
    print(f"✓ IHDP dataset downloaded to {datasets_dir}")
    return datasets_dir


def download_twins():
    """Download Twins mortality dataset."""
    print("Downloading Twins dataset...")
    
    # Note: Twins dataset typically requires preprocessing from raw birth records
    # For now, we'll create a placeholder
    datasets_dir = Path("datasets/twins")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    print("  Note: Twins dataset requires manual download from CDC/NCHS")
    print("  See: https://www.nber.org/research/data/linked-birth-infant-death-cohort-data")
    
    return datasets_dir


def create_dataset_info():
    """Create README with dataset information."""
    readme_content = """# Causal Inference Benchmark Datasets

## IHDP (Infant Health and Development Program)
- **Source**: Randomized experiment (1985-1988)
- **Size**: 747 subjects (139 treated, 608 control)
- **Features**: 25 covariates (6 continuous, 19 binary)
- **Outcome**: Cognitive test score (simulated)
- **Use**: Semi-synthetic benchmark with known ground truth
- **Replications**: 100 datasets in `ihdp/`

### Key Features:
- Birth weight
- Head circumference
- Weeks of gestation  
- Birth order
- Mother's age, education, marital status, race
- Prenatal care indicators

### Treatment Effect:
- True ATE varies by replication (typically 4-8 points)
- Provides ground truth for validation

## Twins
- **Source**: US twin births (1989-1991)
- **Size**: ~10,000 twin pairs
- **Treatment**: Being the heavier twin
- **Outcome**: Mortality in first year
- **Use**: Real observational data for causal inference

### Key Features:
- Gestational age
- Birth weight (each twin)
- Mother/father characteristics
- Pregnancy complications

## Colored MNIST (Semi-Synthetic)
- **Source**: MNIST digits + synthetic confounding
- **Purpose**: Test representation learning for causality
- **Treatment**: Binary (e.g., digit identity)
- **Confounder**: Embedded in color/rotation
- **Outcome**: Synthetic (linear/nonlinear function of U, T)

Generated on-the-fly via `src/data/colored_mnist.py`

## References
1. Hill, J. L. (2011). Bayesian nonparametric modeling for causal inference. JCGS.
2. Louizos, C. et al. (2017). Causal Effect Inference with Deep Latent-Variable Models. NIPS.
3. Alaa, A. & van der Schaar, M. (2017). Bayesian Inference of Individualized Treatment Effects using Multi-task GANs. NIPS.
"""
    
    readme_path = Path("datasets/README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✓ Created dataset documentation: {readme_path}")


if __name__ == '__main__':
    print("Setting up causal inference benchmark datasets...\n")
    
    # Create datasets directory
    Path("datasets").mkdir(exist_ok=True)
    
    # Download datasets
    ihdp_dir = download_ihdp()
    twins_dir = download_twins()
    
    # Create documentation
    create_dataset_info()
    
    print("\n✅ Dataset setup complete!")
    print("\nAvailable datasets:")
    print(f"  - IHDP: {ihdp_dir}")
    print(f"  - Twins: {twins_dir} (manual download required)")
    print(f"  - Colored MNIST: Generated on-the-fly")
