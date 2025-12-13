"""
Batch Optuna Tuning for All Datasets

Runs hyperparameter tuning for SPOCK on multiple datasets and saves
the best parameters to a unified config file.

Usage:
    python tune_all_datasets.py --n_trials 100
    python tune_all_datasets.py --datasets Handwritten NUSwide --n_trials 50
    python tune_all_datasets.py --n_trials 100 --timeout_per_dataset 3600
"""

import os
import sys
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_optuna_tuning import run_optuna_study
from spock.datasets import get_available_datasets


# Default datasets to tune
DEFAULT_DATASETS = [
    'Handwritten',
    'BBCSport', 
    'Caltech101-7',
    'Caltech101-20',
    'NUSwide',
    'Scene15',
]


def tune_all_datasets(
    datasets: list = None,
    n_trials: int = 100,
    metric: str = 'ACC',
    n_eval_runs: int = 3,
    timeout_per_dataset: int = None,
    random_seed: int = 42,
    save_dir: str = './results/optuna',
    config_path: str = './config/tuned_params.json',
):
    """
    Run Optuna tuning on multiple datasets and save unified config.
    
    Parameters
    ----------
    datasets : list
        List of dataset names. If None, uses DEFAULT_DATASETS.
    n_trials : int
        Number of trials per dataset.
    metric : str
        Metric to optimize.
    n_eval_runs : int
        Runs per trial for stability.
    timeout_per_dataset : int
        Max seconds per dataset.
    random_seed : int
        Random seed.
    save_dir : str
        Directory for individual results.
    config_path : str
        Path to save unified config.
    """
    if datasets is None:
        datasets = DEFAULT_DATASETS
    
    # Filter to available datasets
    available = get_available_datasets()
    datasets = [d for d in datasets if d.lower() in [a.lower() for a in available]]
    
    print(f"\n{'='*70}")
    print(f"Batch Optuna Tuning for SPOCK")
    print(f"{'='*70}")
    print(f"Datasets: {datasets}")
    print(f"Trials per dataset: {n_trials}")
    print(f"Metric: {metric}")
    print(f"{'='*70}\n")
    
    # Load existing config if exists
    all_params = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            all_params = json.load(f)
        print(f"Loaded existing config with {len(all_params)} datasets")
    
    # Tune each dataset
    for i, dataset_name in enumerate(datasets):
        print(f"\n[{i+1}/{len(datasets)}] Tuning {dataset_name}...")
        
        try:
            study, best_params = run_optuna_study(
                dataset_name=dataset_name,
                n_trials=n_trials,
                metric=metric,
                n_eval_runs=n_eval_runs,
                timeout=timeout_per_dataset,
                random_seed=random_seed,
                save_dir=save_dir,
            )
            
            # Save to unified config
            all_params[dataset_name.lower()] = {
                'params': best_params,
                'best_value': study.best_value,
                'metric': metric,
                'n_trials': n_trials,
                'timestamp': datetime.now().isoformat(),
            }
            
            # Save after each dataset (in case of interruption)
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(all_params, f, indent=2)
            print(f"Updated config saved to: {config_path}")
            
        except Exception as e:
            print(f"Error tuning {dataset_name}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"Batch Tuning Complete!")
    print(f"{'='*70}")
    print(f"Config saved to: {config_path}")
    print(f"Datasets tuned: {list(all_params.keys())}")
    print(f"{'='*70}\n")
    
    return all_params


def main():
    parser = argparse.ArgumentParser(description='Batch Optuna tuning for all datasets')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to tune (default: all available)')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Trials per dataset')
    parser.add_argument('--metric', type=str, default='ACC',
                        choices=['ACC', 'NMI', 'ARI', 'Purity', 'F1'])
    parser.add_argument('--n_eval_runs', type=int, default=3,
                        help='Eval runs per trial')
    parser.add_argument('--timeout_per_dataset', type=int, default=None,
                        help='Max seconds per dataset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config_path', type=str, default='./config/tuned_params.json',
                        help='Path to save unified config')
    
    args = parser.parse_args()
    
    tune_all_datasets(
        datasets=args.datasets,
        n_trials=args.n_trials,
        metric=args.metric,
        n_eval_runs=args.n_eval_runs,
        timeout_per_dataset=args.timeout_per_dataset,
        random_seed=args.seed,
        config_path=args.config_path,
    )


if __name__ == '__main__':
    main()
