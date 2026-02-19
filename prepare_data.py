import sys
import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import TissueMNISTLoader

# 1. SETUP
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[0]

def initialize_environment():
    """
    Ensures all necessary directories exist and initializes the data loader.
    """
    # Define required paths
    data_root = BASE_DIR / "data" / "raw"

    dirs = [
        BASE_DIR / "results" / "figures",
        BASE_DIR / "results" / "metrics",
        BASE_DIR / "data" / "processed"
    ]
    
    # Create directories
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    logger.info("✓ Directory structure verified.")

    # Load the dataset
    try:
        loader = TissueMNISTLoader(data_root=data_root)
        logger.info(f"✓ TissueMNIST loaded successfully from {data_root}")
        return loader
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

# 2. ANALYTICAL PROTOCOL
def run_analytical_protocol(loader, samples_per_class=500, seed=42):
    """
    Calculates distributions across all splits and identifies imbalance.
    Creates balanced query set.
    """
    np.random.seed(seed)

    train_dist = loader.get_class_distribution('train')
    val_dist = loader.get_class_distribution('val')
    test_dist = loader.get_class_distribution('test')
    
    # Calculate imbalance ratio for training set
    max_count = max(train_dist.values())
    min_count = min(train_dist.values())
    imbalance_ratio = max_count / min_count

    
    # Identify Rare vs Common (Top 3 / Bottom 3)
    sorted_classes = sorted(train_dist.items(), key=lambda x: x[1])
    rare_classes = [int(cls) for cls, _ in sorted_classes[:3]]
    common_classes = [int(cls) for cls, _ in sorted_classes[-3:]]
    
    logger.info(f"Analysis Complete. Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    # Create balanced query set
    labels = loader.test_labels
    balanced_indices = []
    for class_id in range(loader.n_classes):
        class_indices = np.where(labels == class_id)[0]
        n_samples = min(samples_per_class, len(class_indices))
        sampled = np.random.choice(class_indices, size=n_samples, replace=False)
        balanced_indices.extend(sampled)
    
    balanced_indices = np.array(balanced_indices)
    balanced_labels = labels[balanced_indices]
    logger.info(f"✓ Created balanced query set with {len(balanced_indices)} samples.")
    
    return {
        'train_dist': train_dist, 'val_dist': val_dist, 'test_dist': test_dist,
        'imbalance_ratio': imbalance_ratio, 'rare': rare_classes, 'common': common_classes,
        'balanced_indices': balanced_indices,
        'balanced_labels': balanced_labels
    }

# 3. ARTIFACT CREATION
def save_artifacts(loader, results):
    """
    Saves the summary statistics to a CSV file and the metadata to a JSON file.
    """
    processed_path = BASE_DIR / "data" / "processed"
    metrics_path = BASE_DIR / "results" / "metrics"

    # Save Indices and Labels for Balanced Query Set
    np.save(processed_path / 'balanced_query_indices.npy', results['balanced_indices'])
    np.save(processed_path / 'balanced_query_labels.npy', results['balanced_labels'])
    
    # Save Summary CSV (Using Pandas to_string for the console later)
    summary_data = []
    for class_id in sorted(results['train_dist'].keys()):
        class_name = loader.class_names[str(class_id)]
        summary_data.append({
            'Class ID': class_id,
            'Class Name': class_name[:40],
            'Train': results['train_dist'][class_id],
            'Imbalance Factor': f"{(max(results['train_dist'].values()) / results['train_dist'][class_id]):.2f}x",
            'Val': results['val_dist'][class_id],
            'Test': results['test_dist'][class_id],
            'Total': results['train_dist'][class_id] + results['val_dist'][class_id] + results['test_dist'][class_id],
            'Train %': f"{(results['train_dist'][class_id]/sum(results['train_dist'].values()))*100:.2f}%"
    })
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(metrics_path / 'class_statistics.csv', index=False)

    # Save Metadata JSON
    unique, counts = np.unique(results['balanced_labels'], return_counts=True)
    actual_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
    metadata = {
        'dataset_info': {
            'name': 'TissueMNIST v2',
            'citation': 'Zhang, L., et al. (2023). MedMNIST v2. Nature Scientific Data.',
            'database_size': int(len(loader.train_labels)),
            'validation_size': int(len(loader.val_labels)),
            'test_size': int(len(loader.test_labels)),
            'total_size': int(len(loader.train_labels) + len(loader.val_labels) + len(loader.test_labels)),
            'num_classes': loader.n_classes,
            'class_names': loader.class_names,
            'image_shape': list(loader.train_images.shape[1:])
        },
        'imbalance_metrics': {
            'ratio': float(results['imbalance_ratio']),
            'rare_classes': results['rare'],
            'common_classes': results['common'],
            'rare_class_names': {int(c): loader.class_names[str(c)] for c in results['rare']},
            'common_class_names': {int(c): loader.class_names[str(c)] for c in results['common']},
        },
        'evaluation_protocol': {
            'strategy': 'Balanced Queries / Imbalanced Archive',
            'query_size': len(results['balanced_indices']),
            'samples_per_class': actual_distribution,
            'metrics': ['macro_averaged', 'weighted_averaged', 'per_class_breakdown'],
            'rationale': 'Balanced queries ensure fair evaluation; imbalanced database reflects reality'
        },
        'expected_challenges': {
            'rare_class_retrieval': 'Models may struggle with classes having <10K samples',
            'class_confusion': 'Similar tissue types (e.g., tubule segments) may be confused',
            'metric_interpretation': 'Macro vs weighted averages will reveal imbalance effects'
        }
    }

    with open(processed_path / 'split_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("✓ Comprehensive metadata and CSV saved.")
    
    return df_summary

# 4. Visualization Function
def run_visual_protocol(loader, results):
    """
    Generates visualizations for the experimental protocol.
    Figures are saved to ../results/figures/.
    """
    fig_path = BASE_DIR / "results" / "figures"
    logger.info("Generating scientific visualizations...")
    

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    sns.set_palette("husl")
    
    # --- FIGURE 1: THE TISSUE GALLERY ---
    # Purpose: Prove data integrity and show morphological diversity
    fig, axes = plt.subplots(8, 6, figsize=(15, 20))
    fig.suptitle('Figure 1: Morphological Diversity of TissueMNIST Classes', 
                 fontsize=18, fontweight='bold', y=1.02)

    for class_id in range(loader.n_classes):
        samples, _ = loader.get_samples_by_class(class_id, split='train', n_samples=6)
        class_name = loader.class_names[str(class_id)]
        
        for col in range(6):
            ax = axes[class_id, col]
            ax.imshow(samples[col])
            ax.axis('off')
            if col == 0:
                ax.set_title(f"C{class_id}: {class_name}", 
                            fontsize=12, fontweight='bold', loc='left')

    plt.tight_layout()
    plt.savefig(fig_path / 'sample_gallery.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- FIGURE 2: DATABASE VS QUERY DISTRIBUTIONS ---
    # Purpose: Justify the evaluation strategy visually
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    classes = sorted(results['train_dist'].keys())
    class_labels = [f"C{c}" for c in classes]
    
    # Left: The "Realistic" Imbalanced Database
    db_counts = [results['train_dist'][c] for c in classes]
    ax1.bar(class_labels, db_counts, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax1.set_title(f"A: Retrieval Database (Archive)\nImbalance Ratio {results['imbalance_ratio']:.2f}:1", 
                  fontweight='bold')
    ax1.set_ylabel("Image Count")

    # Right: The "Fair" Balanced Query Set
    unique, counts = np.unique(results['balanced_labels'], return_counts=True)
    ax2.bar([f"C{c}" for c in unique], counts, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax2.set_title("B: Evaluation Query Set (Balanced)\nEnsures Fair Per-Class Assessment", 
                  fontweight='bold')
    ax2.set_ylabel("Query Count")

    plt.tight_layout()
    plt.savefig(fig_path / 'distribution_comparison.png', dpi=300)
    plt.close()

    logger.info("✓ Visual Protocol Complete. Figures saved to results/figures/")

# 5. EXECUTION 
def main():
    loader = initialize_environment()
    results = run_analytical_protocol(loader, samples_per_class=500, seed=42)
    run_visual_protocol(loader, results)
    df_summary = save_artifacts(loader, results)
    
    print("\n" + "="*100)
    print("FINAL DATASET SUMMARY")
    print("="*50)
    print(df_summary.to_string(index=False))
    print("="*100)

if __name__ == "__main__":
    main()