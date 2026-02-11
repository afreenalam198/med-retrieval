"""
Test that your environment is set up correctly
"""
import sys
import torch
import torchvision
import numpy as np
import transformers
from qdrant_client import QdrantClient

print("="*60)
print("ENVIRONMENT VERIFICATION")
print("="*60)

# Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")

# PyTorch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ Torchvision version: {torchvision.__version__}")

# Check MPS (Metal Performance Shaders - Apple Silicon GPU)
if torch.backends.mps.is_available():
    print("✓ MPS (Apple Silicon GPU) is available!")
    device = torch.device("mps")
    
    # Test MPS with a simple operation
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    z = x @ y
    print(f"✓ MPS test successful: {z.shape}")
else:
    print("⚠ MPS not available, will use CPU (slower)")
    device = torch.device("cpu")

print(f"✓ Default device: {device}")

# Other libraries
print(f"✓ NumPy version: {np.__version__}")
print(f"✓ Transformers version: {transformers.__version__}")

# Test Qdrant
try:
    client = QdrantClient(":memory:") # Creates temp volatile database in RAM for testing
    print("✓ Qdrant client working")
except Exception as e:
    print(f"✗ Qdrant error: {e}")

# Test medmnist
try:
    import medmnist
    print(f"✓ MedMNIST version: {medmnist.__version__}")
except Exception as e:
    print(f"✗ MedMNIST error: {e}")

print("\n" + "="*60)
print("ENVIRONMENT READY!")
print("="*60)
print("\nYou can now proceed with the project.")
print("Remember to always activate your environment:")
if 'conda' in sys.prefix:
    print("  conda activate medretrieval")
else:
    print("  source venv/bin/activate")