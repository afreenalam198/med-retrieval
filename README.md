# Medical Image Retrieval Benchmark: BiomedCLIP on TissueMNIST

A comprehensive evaluation of vision encoder (BiomedCLIP) for histopathology image retrieval under severe class imbalance, demonstrating fundamental limitations of low-resolution medical imaging for AI-based retrieval systems.

---

## 🎯 Project Overview

### Research Question
How do domain-specific vision encoders perform on medical image retrieval when faced with:
1. **Severe class imbalance** (9:1 ratio between common and rare tissue types)
2. **Low image resolution** (28×28 pixel histopathology samples)

### Key Findings

| Finding | Impact |
|---------|--------|
| **Overall Performance** | 66.0% Recall@5 on 4,000 balanced queries |
| **Class Imbalance Gap** | 34.3 percentage point difference (Rare: 48.4%, Common: 82.7%) |
| **Embedding Quality** | Silhouette Score = -0.025 (near-random clustering) |
| **Resolution Bottleneck** | 28×28 resolution insufficient for tissue discrimination |

**Main Contribution**: Demonstrates that **low image resolution fundamentally limits retrieval performance** even on a generalist biomedical model, and that **class imbalance compounds this limitation** for rare but clinically important tissue types.

---

## 📊 Results Summary

### Overall Retrieval Performance

| Metric | Score |
|--------|-------|
| **Recall@1** | 30.6% |
| **Recall@3** | 54.7% |
| **Recall@5** | 66.0% |
| **Recall@10** | 80.0% |
| **Mean Reciprocal Rank (MRR)** | 0.4549 |

### Per-Class Performance (Recall@5)

| Class | Tissue Type | Database Size | Recall@5 | Category |
|-------|-------------|---------------|----------|----------|
| **0** | Collecting Duct, Connecting Tubule | 53,075 | **92.6%** | Common |
| **1** | Distal Convoluted Tubule | 7,814 | **32.8%** | Rare |
| **2** | Glomerular endothelial cells | 5,866 | **60.4%** | Rare |
| **3** | Interstitial endothelial cells | 15,406 | 69.8% | Medium |
| **4** | Leukocytes | 11,789 | 64.0% | Medium |
| **5** | Podocytes | 7,705 | **52.0%** | Rare |
| **6** | Proximal Tubule Segments | 39,203 | **86.2%** | Common |
| **7** | Thick Ascending Limb | 24,608 | 69.4% | Medium |

**Observation**: Performance strongly correlates with database size. Common classes (>20K samples) achieve 82.7% average Recall@5, while rare classes (<10K samples) achieve only 48.4%.

### Embedding Space Quality
```
Silhouette Score:    -0.0253  (Range: -1 to +1, higher = better separation)
Davies-Bouldin Index: 9.1542  (Range: 0 to ∞, lower = better clustering)

Interpretation: Near-zero silhouette indicates classes are NOT well-separated 
in embedding space, explaining the moderate retrieval performance.
```

**All 8 classes show negative or near-zero silhouette scores**, indicating BiomedCLIP cannot extract discriminative features from 28×28 images.

---

## 🏗️ System Architecture

### Pipeline Overview
```
┌─────────────────┐
│  Query Image    │  28×28 RGB histopathology image
│  (TissueMNIST)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  BiomedCLIP Vision Encoder  │  Microsoft's medical foundation model
│  (ViT-B/16)                 │  Trained on 15M PubMed image-text pairs
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│  512-dim Vector │  L2-normalized embedding (float32)
└────────┬────────┘
         │
         ▼
┌──────────────────────────────┐
│  Qdrant Vector Database      │  HNSW indexing for fast search
│  165,466 embeddings          │  Cosine similarity
└────────┬─────────────────────┘
         │
         ▼
┌─────────────────┐
│  Top-K Results  │  Most similar images by vector distance
└─────────────────┘
```

### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Dataset** | TissueMNIST v2 | Zhang et al. 2023 | Kidney tissue histopathology (236K images) |
| **Model** | BiomedCLIP | microsoft/PubMedBERT_256-vit_base_patch16_224 | Medical vision encoder (512-dim) |
| **Vector DB** | Qdrant | v1.17.0 | Similarity search with HNSW indexing |
| **Deployment** | Docker Compose | 3.8 | Infrastructure orchestration |
| **Framework** | PyTorch | 2.1.0 | Deep learning inference |
| **Hardware** | Apple M2 MacBook Air | 8GB RAM | Resource-constrained setting |
| **Compute** | MPS (Metal) | Apple Silicon | GPU acceleration |

---

## 📁 Project Structure
```
histopathology-retrieval-benchmark/
├── configs/
│   └── biomedclip_config.yaml          # Model & Qdrant configuration
│
├── data/
│   ├── raw/tissuemnist/                # Original dataset (236K images)
│   ├── processed/
│   │   ├── split_info.json             # Metadata & class distributions
│   │   ├── balanced_query_indices.npy  # Stratified evaluation queries
│   │   └── balanced_query_labels.npy
│   └── embeddings/biomedclip/
│       ├── train_embeddings.npy        # 165K × 512 (float32, 323MB)
│       ├── val_embeddings.npy          # 24K × 512 (46MB)
│       └── test_embeddings.npy         # 47K × 512 (92MB)
│
├── src/
│   ├── data_loader.py                  # TissueMNIST utilities
│   ├── embedding_generator.py          # BiomedCLIP inference
│   ├── qdrant_manager.py               # Vector database operations
│   ├── evaluator.py                    # Retrieval metrics
|   ├── evaluation.py                   # Evaluation metrics
|   └── visualization.py                # Visualization utilities for results
│
├── scripts/
│   ├── download_data.py                   # Dataset download script
│   ├── prepare_data.py                    # Metaata preparation script
│   ├── verify_embeddings.py               # Quality checks
│   ├── generate_biomedclip_embeddings.py  # Embedding generation
│   ├── upload_to_qdrant.py                # Database population
│   ├── run_evaluation.py                  # Retrieval benchmark
│   ├── generate_*.py                      # Visualization scripts
│   ├── verify_embeddings.py               # Quality checks
│   └── test_*.py                          # Test scripts
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb  # EDA with visualizations
│
├── results/
│   ├── figures/
│   │   ├── class_distribution.png         # Dataset imbalance
│   │   ├── per_class_heatmap.png          # Performance by class
│   │   ├── rare_vs_common_comparison.png  # Main finding
│   │   ├── tsne_embeddings.png            # Embedding space visualization
│   │   └── confusion_matrix.png           # Error analysis
│   ├── metrics/
│   │   └── biomedclip_evaluation_results.json  # Detailed metrics
│   └── tables/
│       ├── overall_results.csv
│       └── per_class_results.csv
│
├── docker-compose.yaml                 # Qdrant deployment
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker Desktop (for Qdrant)
- 8GB+ RAM
- 30GB disk space

### Installation
```bash
# 1. Clone repository
git clone <your-repo-url>
cd histopathology-retrieval-benchmark

# 2. Create virtual environment
conda create -n medretrieval python=3.10
conda activate medretrieval

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Qdrant vector database
docker-compose up -d

# 5. Verify Qdrant is running
curl http://localhost:6333/health
# Or visit: http://localhost:6333/dashboard
```

### Running the Pipeline
```bash
# Step 1: Download dataset (automatic on first run)
python download_data.py

# Step 2: Generate embeddings (~60 minutes on M2 Mac)
python scripts/generate_biomedclip_embeddings.py

# Step 3: Verify embeddings
python scripts/verify_embeddings.py

# Step 4: Upload to Qdrant (~10 minutes)
python scripts/upload_to_qdrant.py

# Step 5: Run evaluation (~15 minutes)
python scripts/run_evaluation.py

# Step 6: Generate visualizations
python scripts/generate_heatmap.py
python scripts/generate_comparison_plot.py
python scripts/generate_tsne_plot.py
```

---

## 📈 Detailed Analysis

### 1. Class Imbalance Impact

**Dataset Distribution:**
- **Imbalance Ratio**: 9.05:1
- **Most common**: Class 0 (Collecting Duct) - 53,075 samples (32%)
- **Rarest**: Class 2 (Glomerular endothelial) - 5,866 samples (3.5%)

**Performance Gap:**
```
Common Classes (top 3):  82.7% Recall@5
Rare Classes (bottom 3): 48.4% Recall@5
Gap:                     34.3 percentage points
```

**Interpretation**: Model performs dramatically better on frequently-seen tissue types, creating a **34-point disadvantage for rare but clinically critical pathologies**.

---

### 2. Resolution Bottleneck

**TissueMNIST Images:**
- Original resolution: **28×28 pixels**
- Upsampled for BiomedCLIP: **224×224 pixels** (8× interpolation)
- Result: Blurry images with minimal tissue detail

**Evidence from Clustering Analysis:**
```
Silhouette Score: -0.0253
├─ Class 0: -0.0234  ❌ Highly Confused
├─ Class 1: -0.0155  ❌ Highly Confused
├─ Class 2:  0.0068  ⚠️ Overlapping
├─ Class 3: -0.0360  ❌ Highly Confused
├─ Class 4: -0.0289  ❌ Highly Confused
├─ Class 5: -0.0263  ❌ Highly Confused
├─ Class 6: -0.0320  ❌ Highly Confused
└─ Class 7: -0.0470  ❌ Highly Confused
```

**All classes show negative or near-zero silhouette**, indicating BiomedCLIP cannot extract discriminative features from low-resolution images.

**Comparison to Literature:**
- Full-resolution pathology: 1000×1000+ pixels
- Clear visualization of cell structures, nuclei, tissue patterns
- Our dataset: 28×28 pixels (96%+ resolution loss)

---

### 3. Methodology Strengths

**Addressing Imbalance:**
1. **Kept database imbalanced** (realistic medical archive)
2. **Created balanced query set** (500 samples per class = 4,000 total)
3. **Reported both macro and weighted metrics**:
   - Macro-averaged: Treats all classes equally
   - Weighted-averaged: Reflects database distribution

**Result**: Fair evaluation that reveals true performance disparities

**Evaluation Rigor:**
- Stratified sampling ensures each class tested equally
- Multiple metrics (Recall@K, MRR)
- Per-class breakdown identifies strengths/weaknesses
- Statistical analysis (silhouette, Davies-Bouldin) validates findings

---

### Limitations

1. **Dataset Resolution**: 28×28 is far below clinical standards (1000×1000+)
2. **No Fine-tuning**: Used pre-trained BiomedCLIP without adaptation
3. **Single Model**: Would benefit from CLIP/ResNet/DINOv2 comparison
4. **Synthetic Task**: TissueMNIST is simplified vs. real diagnostic workflows

### Future Work

1. **Higher Resolution Data**
   - Evaluate on full-resolution histopathology (e.g., Camelyon16: 100K×100K)
   - Expected: Dramatic improvement in embedding separability

2. **Model Comparison**
   - Add OpenAI CLIP (general-purpose baseline)
   - Add ResNet50 (supervised learning baseline)
   - Hypothesis: BiomedCLIP will have smallest rare/common gap

3. **Imbalance Mitigation**
   - Test retrieval-time reweighting (boost rare classes)
   - Implement query expansion using text encoder
   - Fine-tune with class-balanced sampling

4. **Clinical Validation**
   - Collaborate with pathologists for ground truth
   - Test on real diagnostic cases
   - Measure clinical utility beyond accuracy metrics

---

## 🎓 My Learnings

**Concepts Demonstrated:**
- ✅ Embedding spaces and similarity search
- ✅ Vector databases (Qdrant, HNSW indexing)
- ✅ Class imbalance in real-world datasets
- ✅ Evaluation methodology (balanced queries, macro averaging)
- ✅ Clustering quality metrics (silhouette, Davies-Bouldin)

**Practical Lessons:**
- ✅ Domain-specific models don't guarantee success
- ✅ Data quality matters more than model sophistication
- ✅ Rare classes require special evaluation attention
- ✅ Negative results are valuable when well-analyzed

---

## 📊 Computational Requirements

**Optimizations for 8GB RAM:**
- Batch processing (32 images/batch)
- Memory-mapped Qdrant storage
- Progressive embedding generation with checkpoints
- Apple MPS GPU acceleration (65 img/sec)


---

**Dataset Citation:**
```bibtex
@article{medmnistv2,
  title={MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
  author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={41},
  year={2023},
  publisher={Nature Publishing Group}
}
```

**Model Citation:**
```bibtex
@article{biomedclip,
  title={BiomedCLIP: A multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs},
  author={Zhang, Sheng and Xu, Yanbo and Usuyama, Naoto and Bagga, Jaspreet and Tinn, Robert and Preston, Sam and Rao, Rajesh and Wei, Mu and Valluri, Naveen and Wong, Cliff and others},
  journal={arXiv preprint arXiv:2303.00915},
  year={2023}
}
```

