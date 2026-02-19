"""
Download TissueMNIST dataset from MedMNIST
"""
import numpy as np
import os
from pathlib import Path
import medmnist
from medmnist import INFO

def download_tissuemnist():
    """Download and verify TissueMNIST dataset"""
    
    print("="*70)
    print("DOWNLOADING TISSUEMNIST DATASET")
    print("="*70)
    
    # Get TissueMNIST info
    data_flag = 'tissuemnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Create data directory
    data_dir = Path('./data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading to: {data_dir.absolute()}")
    print("This may take a few minutes (dataset is ~180MB)...\n")
    
    # Download each split
    print("Downloading training set...")
    train_dataset = DataClass(split='train', download=True, root='./data/raw')
    
    print("Downloading validation set...")
    val_dataset = DataClass(split='val', download=True, root='./data/raw')
    
    print("Downloading test set...")
    test_dataset = DataClass(split='test', download=True, root='./data/raw')
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  • Training set:   {len(train_dataset):,} images")
    print(f"  • Validation set: {len(val_dataset):,} images")
    print(f"  • Test set:       {len(test_dataset):,} images")
    print(f"  • Total:          {len(train_dataset) + len(val_dataset) + len(test_dataset):,} images")
    print(f"  • Classes:        {len(info['label'])} tissue types")
    print(f"  • Resolution:     28×28 pixels (RGB)")
    
    print(f"\n🏷️  Class Labels:")
    for idx, (key, value) in enumerate(info['label'].items()):
        print(f"  Class {key}: {value}")
    
    # Verify download
    print(f"\n✓ Files saved to: {data_dir.absolute()}")
    
    # Check file sizes
    npz_file = data_dir / 'tissuemnist.npz'
    if npz_file.exists():
        size_mb = npz_file.stat().st_size / (1024 * 1024)
        print(f"✓ Dataset file size: {size_mb:.1f} MB")
    
    print("\n" + "="*70)
    print("You can now proceed to data analysis!")
    print("="*70)

if __name__ == "__main__":
    download_tissuemnist()