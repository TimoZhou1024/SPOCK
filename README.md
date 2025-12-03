# SPOCK: Scalable and Structure-Preserving Optimal Transport based Clustering with Kernel-density-estimation

A Python implementation of the SPOCK algorithm for multi-view clustering, achieving **96.05% ACC** on the Handwritten dataset.

## Overview

SPOCK is a three-phase framework for scalable multi-view clustering with near-linear complexity:

1. **Phase 1: Structure-Preserving Sparse Feature Selection** — $O(NDd)$
   - Randomized SVD for efficient dimensionality reduction
   - Sparse Laplacian regularization for local manifold preservation
   - Row-normalization for stable distance computation

2. **Phase 2: Density-Aware View Weighting** — $O(Nk\log N)$
   - $k$-NN based Kernel Density Estimation for view quality assessment
   - Adaptive weighting combining density uniformity and feature compactness
   - Parameter $\mu$ controls the trade-off (default: 0.7)

3. **Phase 3: OT-Enhanced Spectral Clustering** — $O(NMT)$
   - Landmark-based Sinkhorn Optimal Transport ($M \ll N$ landmarks)
   - Additive graph enhancement combining local $k$-NN and global OT information
   - Sparse spectral embedding via ARPACK

**Total Complexity:** $O(N \log N)$ — near-linear scalability!

## Installation

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the project:

```bash
cd SPOCK

# Create virtual environment and install dependencies
uv sync

# Install with development dependencies
uv sync --dev
```

### Using pip

```bash
pip install -e .
```

## Quick Start

```python
from spock import SPOCK
from spock.datasets import load_handwritten
from spock.evaluation import evaluate_clustering

# Load data
dataset = load_handwritten()

# Run SPOCK (Spectral mode for best accuracy)
model = SPOCK(
    n_clusters=dataset.n_clusters,
    use_spectral=True,  # Use spectral clustering
    verbose=True,
    random_state=42
)

labels = model.fit_predict(dataset.views)

# Evaluate
results = evaluate_clustering(dataset.labels, labels)
print(f"ACC: {results['ACC']:.4f}, NMI: {results['NMI']:.4f}")
# Expected: ACC: 0.9605, NMI: 0.9172
```

Or run the demo:
```bash
uv run python demo.py
```

## Hyperparameters

### Core Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `n_clusters` | Number of clusters | Required | - |
| `k_neighbors` | KNN neighbors for graph construction | 10 | [5, 50] |
| `proj_dim` | Projection dimension | 100 | [50, 200] |
| `n_landmarks` | OT landmarks ($M$) | 500 | [100, 1000] |
| `use_spectral` | Use spectral clustering vs KMeans | False | - |

### Phase 2: View Weighting

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `mu` | Density uniformity vs compactness trade-off | 0.7 | [0, 1] |

$$w_v = \frac{\mu \cdot q_{\text{density}}^{(v)} + (1-\mu) \cdot q_{\text{compact}}^{(v)}}{\sum_{u} (\mu \cdot q_{\text{density}}^{(u)} + (1-\mu) \cdot q_{\text{compact}}^{(u)})}$$

### Phase 3: Graph Enhancement

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `gamma` | OT bonus strength | 0.06 | [0, 0.2] |
| `tau` | OT similarity threshold | 0.5 | [0, 1] |
| `sinkhorn_reg` | Entropy regularization ($\varepsilon$) | 0.1 | [0.01, 1.0] |

$$W'_{ij} = W_{ij} \cdot \rho_{ij} + \gamma \cdot [\mathbf{t}_i^\top \mathbf{t}_j - \tau]_+$$

### Advanced Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `alpha` | Laplacian regularization weight | 1.0 |
| `rff_dim` | Random Fourier Features dimension | 256 |
| `max_iter` | ADMM iterations | 50 |
| `sinkhorn_iter` | Sinkhorn iterations | 100 |

## Experiments

### Main Experiments

```bash
# Run on all datasets
uv run python experiments/run_experiments.py --dataset all --n_runs 10

# Run on specific dataset
uv run python experiments/run_experiments.py --dataset Handwritten --n_runs 10
```

### Scalability Analysis

```bash
uv run python experiments/run_scalability.py --max_samples 50000
```

### Ablation Study

```bash
uv run python experiments/run_ablation.py --dataset Handwritten --n_runs 10
```

### Parameter Sensitivity

```bash
uv run python experiments/run_sensitivity.py --dataset Handwritten --param mu
uv run python experiments/run_sensitivity.py --dataset Handwritten --param gamma
```

## Datasets

### Supported Datasets

| Dataset | Samples | Classes | Views | Dimensions | Status |
|---------|---------|---------|-------|------------|--------|
| **Handwritten** | 2000 | 10 | 6 | [216, 76, 64, 6, 240, 47] | ✅ Auto |
| **BBCSport** | 544 | 5 | 2 | [3183, 3203] | ✅ Ready |
| **COIL-20** | 1440 | 20 | multi | varies | Manual |
| **Caltech101-7** | 1474 | 7 | 6 | varies | Manual |
| **Caltech101-20** | 2386 | 20 | 6 | varies | Manual |
| **Scene15** | 4485 | 15 | 3 | varies | Manual |

### Loading Datasets

```python
from spock.datasets import (
    load_handwritten,
    load_bbcsport,
    load_caltech101,
    load_scene15,
)

# Auto-download
dataset = load_handwritten()

# BBCSport (from MatrixMarket files in data/bbcsport/)
dataset = load_bbcsport()

# Manual download required
dataset = load_caltech101(n_classes=7)
```

## Project Structure

```
SPOCK/
├── spock/
│   ├── core/
│   │   └── spock_algorithm.py    # Core SPOCK implementation
│   ├── datasets/
│   │   ├── loaders.py            # Dataset loaders (including BBCSport MTX)
│   │   └── download.py           # Auto-download utilities
│   ├── evaluation/
│   │   └── metrics.py            # ACC, NMI, ARI, Purity, F1
│   └── baselines/
│       └── methods.py            # Baseline methods for comparison
├── experiments/
│   ├── run_experiments.py        # SOTA comparison
│   ├── run_ablation.py           # Ablation study
│   ├── run_sensitivity.py        # Parameter sensitivity
│   └── run_scalability.py        # Scalability tests
├── data/
│   ├── handwritten.mat           # Handwritten digits
│   └── bbcsport/                 # BBCSport (MatrixMarket format)
├── results/                      # Experiment results
├── demo.py                       # Quick demo
├── main.tex                      # Paper (ICML 2026)
└── pyproject.toml               # Project configuration
```

## Baseline Methods

SPOCK is compared against several multi-view clustering baselines:

| Method | Description |
|--------|-------------|
| **Concat+KMeans** | Concatenate views + K-Means |
| **Concat+Spectral** | Concatenate views + Spectral Clustering |
| **Best-View** | Best single-view Spectral Clustering |
| **Co-Reg** | Co-Regularized Spectral Clustering |
| **MV-KMeans** | Multi-View K-Means |
| **MVSC** | Multi-View Spectral Clustering |
| **LMvSC** | Large-scale Multi-View Subspace Clustering |
| **MLAN** | Multi-View Learning with Adaptive Neighbors |

## Evaluation Metrics

- **ACC**: Clustering Accuracy (with Hungarian matching)
- **NMI**: Normalized Mutual Information
- **ARI**: Adjusted Rand Index
- **Purity**: Cluster Purity
- **F1**: Macro F1-score

<!-- ## Citation

```bibtex
@inproceedings{spock2026,
  title={SPOCK: Scalable and Structure-Preserving Optimal Transport based Clustering 
         with Kernel-density-estimation for Imperfect Multi-View Data},
  author={...},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

MIT License -->
