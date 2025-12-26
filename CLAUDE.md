# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPOCK (Scalable and Structure-Preserving Optimal Transport based Clustering with Kernel-density-estimation) is a Python implementation of a three-phase multi-view clustering algorithm with near-linear O(N log N) complexity.

## Common Commands

### Environment Setup
```bash
# Using uv (recommended)
uv sync              # Install dependencies
uv sync --dev        # Install with dev dependencies
uv sync --all-extras # Install all optional dependencies (torch, faiss, optuna, etc.)

# Using pip
pip install -e .
```

### Running Experiments
```bash
# Quick demo
uv run python demo.py

# Main experiments
uv run python experiments/run_experiments.py --dataset Handwritten --n_runs 10
uv run python experiments/run_experiments.py --dataset Handwritten --spock_only     # SPOCK only
uv run python experiments/run_experiments.py --dataset Handwritten --include_scalable  # Include scalable baselines
uv run python experiments/run_experiments.py --dataset Handwritten --use_tuned      # Use Optuna-tuned params

# Hyperparameter tuning
uv run python experiments/run_optuna_tuning.py --dataset Handwritten --n_trials 100
uv run python experiments/tune_all_datasets.py --n_trials 100

# Other experiments
uv run python experiments/run_ablation.py --dataset Handwritten --n_runs 10
uv run python experiments/run_sensitivity.py --dataset Handwritten --param mu
uv run python experiments/run_scalability.py --max_samples 50000
```

### Code Quality
```bash
uv run black spock/           # Format code
uv run isort spock/           # Sort imports
uv run flake8 spock/          # Lint
uv run mypy spock/            # Type check
uv run pytest                 # Run tests (testpaths: tests/)
```

## Architecture

### Core Algorithm (`spock/core/spock_algorithm.py`)
The SPOCK class implements a three-phase clustering framework:
- **Phase 1**: Structure-Preserving Sparse Feature Selection via ADMM optimization
- **Phase 2**: Density-Aware View Weighting using RFF-accelerated KDE
- **Phase 3**: OT-Enhanced Spectral Clustering with Nyström approximation

Key parameters: `n_clusters`, `k_neighbors`, `proj_dim`, `n_landmarks`, `mu` (view weighting), `gamma`/`tau` (OT bonus), `use_spectral`

### Module Structure
- `spock/core/` - Core SPOCK algorithm and ablation variants
- `spock/datasets/` - Dataset loaders (`load_dataset()`, `load_handwritten()`, etc.)
- `spock/evaluation/` - Metrics (ACC, NMI, ARI, Purity, F1)
- `spock/baselines/methods.py` - Traditional baselines (ConcatKMeans, MVSC, Co-Reg, etc.)
- `spock/baselines/scalable_methods.py` - Scalable SOTA (LMVSC, SMVSC, FMCNOF, EOMSC-CA, BMVC)
- `spock/baselines/external/` - External methods requiring git clone (SCMVC, EFIMVC, ALPC)

### Data Flow
1. Load dataset via `load_dataset('DatasetName')` or specific loader
2. Optionally normalize: `dataset.normalize('standard')`
3. Create SPOCK model with parameters
4. Call `model.fit_predict(dataset.views)` - returns cluster labels
5. Evaluate with `evaluate_clustering(true_labels, predicted_labels)`

### Supported Datasets
Handwritten, BBCSport, NUS-WIDE, COIL-20, Caltech101-7, Caltech101-20, Scene15

Data files go in `data/` directory (auto-downloaded for some datasets).

### Configuration
- `config/tuned_params.json` - Optuna-tuned hyperparameters per dataset
- Experiment results saved to `results/`
