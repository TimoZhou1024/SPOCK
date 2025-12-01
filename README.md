# SPOCK: Scalable and Structure-Preserving Optimal Transport based Clustering with Kernel-density-estimation

A Python implementation of the SPOCK algorithm for multi-view clustering.

## Overview

SPOCK is a three-phase framework for scalable multi-view clustering:

1. **Phase 1: Unsupervised Structure-Preserving Sparse Feature Selection**
   - Learns projection matrices that preserve local manifold and self-expression structure
   - Uses L2,1 regularization for row-sparse feature selection
   - Optimized via ADMM

2. **Phase 2: Density-Aware Graph Alignment**
   - Uses Random Fourier Features (RFF) for O(N) kernel density estimation
   - Constructs density-aware graphs filtering low-density regions
   - Aligns views using RFF-accelerated Optimal Transport

3. **Phase 3: Final Clustering**
   - Uses Nyström approximation for O(NM²) spectral embedding
   - Applies K-Means on the spectral embedding

## Installation

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it first:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the project:

```bash
# Clone and enter the project directory
cd SPOCK

# Create virtual environment and install dependencies
uv sync

# Install with optional dependencies (GPU, FAISS, etc.)
uv sync --extra full

# Install with development dependencies
uv sync --dev
```

### Using pip (Alternative)

```bash
pip install -e .

# Or with optional dependencies
pip install -e ".[full]"
```

### Legacy requirements.txt

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from spock import SPOCK
from spock.datasets import load_dataset
from spock.evaluation import evaluate_clustering

# Load data
dataset = load_dataset('Handwritten')
dataset.normalize('standard')

# Run SPOCK
model = SPOCK(
    n_clusters=dataset.n_clusters,
    alpha=1.0,
    beta=0.1,
    lambda_l21=0.01,
    k_neighbors=10,
    random_state=42
)

labels = model.fit_predict(dataset.views)

# Evaluate
results = evaluate_clustering(dataset.labels, labels)
print(f"ACC: {results['ACC']:.4f}, NMI: {results['NMI']:.4f}")
```

Or run the demo:
```bash
# Using uv
uv run python demo.py

# Or if environment is activated
python demo.py
```

## Experiments

### Main Experiments (SOTA Comparison)

Compare SPOCK with baseline methods on multiple datasets:

```bash
# Run on all datasets
uv run python experiments/run_experiments.py --dataset all --n_runs 10

# Run on a specific dataset
uv run python experiments/run_experiments.py --dataset Handwritten --n_runs 10
```

### Ablation Study

Analyze the contribution of each component:

```bash
uv run python experiments/run_ablation.py --dataset Handwritten --n_runs 10
```

### Parameter Sensitivity

Analyze hyperparameter sensitivity:

```bash
# Test all parameters
uv run python experiments/run_sensitivity.py --dataset Handwritten --param all

# Test specific parameter
uv run python experiments/run_sensitivity.py --dataset Handwritten --param alpha
```

### Scalability Analysis

Test computational efficiency:

```bash
uv run python experiments/run_scalability.py --max_samples 50000
```

## Project Structure

```
SPOCK/
├── spock/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── spock_algorithm.py    # Core SPOCK implementation
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── loaders.py            # Dataset loaders
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py            # Clustering metrics
│   └── baselines/
│       ├── __init__.py
│       └── methods.py            # Baseline methods
├── experiments/
│   ├── run_experiments.py        # Main experiments
│   ├── run_ablation.py           # Ablation study
│   ├── run_sensitivity.py        # Parameter sensitivity
│   └── run_scalability.py        # Scalability tests
├── demo.py                       # Quick demo
├── config.yaml                   # Configuration
├── requirements.txt              # Dependencies
└── README.md
```

## Datasets

### Automatic Download

Some datasets can be downloaded automatically:

```bash
# List all available datasets
uv run python -m spock.datasets.download --list

# Download a specific dataset (e.g., UCI Handwritten Digits)
uv run python -m spock.datasets.download --dataset handwritten

# Check download status
uv run python -m spock.datasets.download --status
```

### Manual Download

For datasets that require manual download:

```bash
# Show download instructions
uv run python -m spock.datasets.download --manual bbcsport
uv run python -m spock.datasets.download --manual reuters
```

### Supported Datasets

| Dataset | Samples | Classes | Views | Status |
|---------|---------|---------|-------|--------|
| **Handwritten** | 2000 | 10 | 6 | ✅ Auto-download |
| **COIL-20** | 1440 | 20 | multi | ✅ Auto-download |
| **BBCSport** | 544 | 5 | 2 | Manual |
| **Caltech101-7** | 1474 | 7 | 6 | Manual |
| **Caltech101-20** | 2386 | 20 | 6 | Manual |
| **Scene15** | 4485 | 15 | 3 | Manual |
| **Reuters** | 18758 | 6 | 5 | Manual |
| **MSRC-v1** | 210 | 7 | 6 | Manual |
| **NUS-WIDE** | 30000 | 31 | 5 | Manual |

Place downloaded `.mat` files in `./data/` directory.

## Evaluation Metrics

- **ACC**: Clustering Accuracy (Hungarian matching)
- **NMI**: Normalized Mutual Information
- **ARI**: Adjusted Rand Index
- **Purity**: Cluster Purity
- **F1**: Macro F1-score

## Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `alpha` | Self-expression weight | 1.0 |
| `beta` | S sparsity (L1) weight | 0.1 |
| `lambda_l21` | P sparsity (L2,1) weight | 0.01 |
| `k_neighbors` | KNN neighbors | 10 |
| `proj_dim` | Projection dimension | 100 |
| `rff_dim` | RFF dimension | 256 |
| `n_landmarks` | Nyström landmarks | 500 |

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{spock2026,
  title={SPOCK: Scalable and Structure-Preserving Optimal Transport based Clustering with Kernel-density-estimation for Imperfect Multi-View Data},
  author={...},
  booktitle={ICML},
  year={2026}
}
```

## Development

### Setting up Development Environment

```bash
# Install with dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black spock/ experiments/
uv run isort spock/ experiments/

# Type checking
uv run mypy spock/
```

### Project Commands

```bash
# Run demo
uv run spock-demo

# Run experiments (after installation)
uv run spock-experiment --dataset Handwritten
uv run spock-ablation --dataset Handwritten
uv run spock-sensitivity --dataset Handwritten --param alpha
uv run spock-scalability --max_samples 10000
```

## License

MIT License
