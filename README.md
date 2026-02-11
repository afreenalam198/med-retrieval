# Histopathology Retrieval Benchmark

A comprehensive benchmark for histopathology image retrieval using various embedding models including BiomedCLIP, OpenAI CLIP, UNI, and ResNet50.

## Project Structure

```
histopathology-retrieval/
├── data/
│   ├── raw/                    # Original MedMNIST downloads
│   ├── processed/              # Train/val/test splits
│   └── embeddings/             # Saved embeddings for each model
├── models/
│   ├── biomedclip/            # BiomedCLIP checkpoint cache
│   ├── clip/                  # OpenAI CLIP cache
│   ├── uni/                   # UNI model (if accessible)
│   └── resnet/                # ResNet50 checkpoint
├── qdrant_storage/            # Vector DB persistence
├── configs/
│   └── experiment_config.yaml  # All hyperparameters
├── src/
│   ├── data_loader.py
│   ├── embedding_generator.py
│   ├── evaluation.py
│   └── visualization.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── results/
│   ├── metrics/               # JSON files with numbers
│   ├── figures/               # Plots for paper
│   └── logs/                  # Experiment logs
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Configure experiments in `configs/experiment_config.yaml`
2. Load your data using `src/data_loader.py`
3. Generate embeddings with `src/embedding_generator.py`
4. Evaluate retrieval performance with `src/evaluation.py`
5. Visualize results using `src/visualization.py`

## Models

- **BiomedCLIP**: Vision-language model trained on biomedical data
- **OpenAI CLIP**: General-purpose vision-language model
- **UNI**: Universal pathology foundation model
- **ResNet50**: Convolutional neural network baseline

## Results

Results are saved in the `results/` directory:
- `metrics/`: Numerical evaluation metrics (JSON format)
- `figures/`: Visualization plots for analysis
- `logs/`: Experiment logs and runtime information

## License

MIT License
