"""
Data loading utilities for TissueMNIST
"""
import numpy as np
from pathlib import Path
import medmnist
from medmnist import INFO
from typing import Tuple, Dict

class TissueMNISTLoader:
    """Loader for TissueMNIST dataset with convenient access methods"""
    
    def __init__(self, data_root: str = './data/raw'):
        """
        Initialize the data loader
        
        Args:
            data_root: Path to raw data directory
        """
        self.data_root = Path(data_root)
        self.data_flag = 'tissuemnist'
        self.info = INFO[self.data_flag]
        self.DataClass = getattr(medmnist, self.info['python_class'])
        
        print("Loading TissueMNIST dataset...")
        
        # Load all splits (download=False assumes data is already downloaded)
        self.train_dataset = self.DataClass(split='train', download=False, root=str(self.data_root))
        self.val_dataset = self.DataClass(split='val', download=False, root=str(self.data_root))
        self.test_dataset = self.DataClass(split='test', download=False, root=str(self.data_root))
        
        # Extract images and labels as numpy arrays for efficient processing
        print("  Extracting training data...")
        self.train_images, self.train_labels = self._extract_data(self.train_dataset)
        
        print("  Extracting validation data...")
        self.val_images, self.val_labels = self._extract_data(self.val_dataset)
        
        print("  Extracting test data...")
        self.test_images, self.test_labels = self._extract_data(self.test_dataset)
        
        self.class_names = self.info['label']
        self.n_classes = len(self.class_names)
        
        print(f"✓ Loaded {len(self.train_labels):,} train, {len(self.val_labels):,} val, {len(self.test_labels):,} test images")
        
    def _extract_data(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract images and labels from MedMNIST dataset
        
        Args:
            dataset: MedMNIST dataset object
            
        Returns:
            images: numpy array of shape (N, 28, 28, 3)
            labels: numpy array of shape (N,)
        """
        images = []
        labels = []
        
        for img, label in dataset:
            images.append(np.array(img))
            labels.append(label[0])  # Labels come as [class_id], extract the int
            
        return np.array(images), np.array(labels)
    
    def get_split_info(self) -> Dict:
        """
        Return comprehensive information about dataset splits
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'dataset_name': 'TissueMNIST',
            'train_size': len(self.train_labels),
            'val_size': len(self.val_labels),
            'test_size': len(self.test_labels),
            'total_size': len(self.train_labels) + len(self.val_labels) + len(self.test_labels),
            'n_classes': self.n_classes,
            'class_names': self.class_names,
            'image_shape': self.train_images.shape[1:],  # (28, 28, 3)
            'image_dtype': str(self.train_images.dtype)
        }
    
    def get_class_distribution(self, split: str = 'train') -> Dict[int, int]:
        """
        Get distribution of classes in a specific split
        
        Args:
            split: One of 'train', 'val', or 'test'
            
        Returns:
            Dictionary mapping class_id -> count
        """
        if split == 'train':
            labels = self.train_labels
        elif split == 'val':
            labels = self.val_labels
        elif split == 'test':
            labels = self.test_labels
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        unique, counts = np.unique(labels, return_counts=True)
        distribution = {int(cls): int(count) for cls, count in zip(unique, counts)}
        return distribution
    
    def get_samples_by_class(self, class_id: int, split: str = 'train', n_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get sample images from a specific class
        
        Args:
            class_id: Class to sample from (0-7)
            split: Which split to use
            n_samples: Number of samples to return
            
        Returns:
            images: Array of shape (n_samples, 28, 28, 3)
            indices: Array of indices where these images were found
        """
        if split == 'train':
            images, labels = self.train_images, self.train_labels
        elif split == 'val':
            images, labels = self.val_images, self.val_labels
        elif split == 'test':
            images, labels = self.test_images, self.test_labels
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Find indices of this class
        class_indices = np.where(labels == class_id)[0]
        
        if len(class_indices) == 0:
            raise ValueError(f"No samples found for class {class_id} in {split} set")
        
        # Sample randomly
        n_samples = min(n_samples, len(class_indices))
        sampled_indices = np.random.choice(class_indices, size=n_samples, replace=False)
        
        return images[sampled_indices], sampled_indices


# Test the loader
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING TISSUEMNIST LOADER")
    print("="*70 + "\n")
    
    loader = TissueMNISTLoader()
    
    info = loader.get_split_info()
    print(f"\n Dataset Information:")
    print(f"  • Dataset: {info['dataset_name']}")
    print(f"  • Training samples:   {info['train_size']:,}")
    print(f"  • Validation samples: {info['val_size']:,}")
    print(f"  • Test samples:       {info['test_size']:,}")
    print(f"  • Total samples:      {info['total_size']:,}")
    print(f"  • Number of classes:  {info['n_classes']}")
    print(f"  • Image shape:        {info['image_shape']}")
    print(f"  • Image dtype:        {info['image_dtype']}")
    
    print(f"\n  Class Names:")
    for class_id, name in info['class_names'].items():
        print(f"  Class {class_id}: {name}")
    
    print(f"\n Training Set Class Distribution:")
    train_dist = loader.get_class_distribution('train')
    total = sum(train_dist.values())
    for class_id in sorted(train_dist.keys()):
        count = train_dist[class_id]
        percentage = (count / total) * 100
        class_name = loader.class_names[str(class_id)]
        print(f"  Class {class_id} ({class_name[:30]:30s}): {count:6,} ({percentage:5.2f}%)")
    
    # Calculate imbalance
    max_count = max(train_dist.values())
    min_count = min(train_dist.values())
    imbalance_ratio = max_count / min_count
    print(f"\n  Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
    print(f"  • Most common class: {max_count:,} samples")
    print(f"  • Rarest class:      {min_count:,} samples")
    
    print("\n" + "="*70)
    print("✓ DATA LOADER TEST PASSED")
    print("="*70)