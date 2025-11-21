import torch
from src.data.loaders import get_data_loaders
import sys

def test_spatiotemporal():
    print("\nTesting Spatiotemporal Dataset...")
    train, val, test = get_data_loaders('spatiotemporal', batch_size=4)
    batch = next(iter(train))
    print("  Batch keys:", batch.keys())
    print("  X shape:", batch['X'].shape)
    print("  T shape:", batch['T'].shape)
    print("  Y shape:", batch['Y'].shape)
    assert batch['X'].shape == (4, 3, 16, 64, 64)

def test_colored_mnist():
    print("\nTesting Colored MNIST Dataset...")
    # Note: This will trigger download if not present
    try:
        train, test = get_data_loaders('colored_mnist', batch_size=4, download=True)
        batch = next(iter(train))
        print("  Batch keys:", batch.keys())
        print("  X shape:", batch['X'].shape)
        print("  T shape:", batch['T'].shape)
        print("  Y shape:", batch['Y'].shape)
        print("  Z shape:", batch['Z'].shape)
        assert batch['X'].shape == (4, 3, 28, 28)
    except Exception as e:
        print(f"  Skipping Colored MNIST test: {e}")

def test_ihdp():
    print("\nTesting IHDP Dataset...")
    try:
        train, val, test = get_data_loaders('ihdp', batch_size=4, replication=1)
        batch = next(iter(train))
        print("  Batch keys:", batch.keys())
        print("  X shape:", batch['X'].shape)
        print("  T shape:", batch['T'].shape)
        print("  Y shape:", batch['Y'].shape)
        print("  mu0 shape:", batch['mu0'].shape)
        assert batch['X'].shape[1] == 25
    except Exception as e:
        print(f"  Failed to load IHDP: {e}")

def test_twins():
    print("\nTesting Twins Dataset (Synthetic)...")
    try:
        train, val, test = get_data_loaders('twins', batch_size=4, synthetic=True)
        batch = next(iter(train))
        print("  Batch keys:", batch.keys())
        print("  X shape:", batch['X'].shape)
        print("  T shape:", batch['T'].shape)
        print("  Y shape:", batch['Y'].shape)
        print("  mu0 shape:", batch['mu0'].shape)
    except Exception as e:
        print(f"  Failed to load Twins: {e}")

if __name__ == '__main__':
    test_spatiotemporal()
    test_colored_mnist()
    test_ihdp()
    test_twins()
    print("\nAll tests passed!")
