"""
Main Experiment Script for SPOCK

Compares SPOCK with baseline methods on multiple datasets.
Usage:
    python run_experiments.py --dataset all --n_runs 10
    python run_experiments.py --dataset Handwritten --n_runs 5
    python run_experiments.py --dataset Handwritten --use_tuned  # Use Optuna-tuned params
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spock import SPOCK
from spock.datasets import load_dataset, get_available_datasets
from spock.baselines import get_baseline_methods
from spock.evaluation import evaluate_clustering, MetricTracker


# Path to tuned parameters config
TUNED_PARAMS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'config', 'tuned_params.json'
)


def load_tuned_params(dataset_name: str) -> dict:
    """
    Load Optuna-tuned parameters for a dataset.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
        
    Returns
    -------
    params : dict or None
        Tuned parameters if available, else None.
    """
    if not os.path.exists(TUNED_PARAMS_PATH):
        return None
    
    try:
        with open(TUNED_PARAMS_PATH, 'r') as f:
            all_params = json.load(f)
        
        key = dataset_name.lower()
        if key in all_params:
            return all_params[key]['params']
        return None
    except Exception as e:
        print(f"Warning: Could not load tuned params: {e}")
        return None


def get_default_spock_params(dataset, X_views) -> dict:
    """
    Get default SPOCK parameters based on dataset characteristics.
    """
    n_samples = dataset.n_samples
    min_dim = min(v.shape[1] for v in X_views)
    
    return {
        'alpha': 1.0,
        'beta': 0.1,
        'lambda_l21': 0.01,
        'k_neighbors': 10,
        'proj_dim': min(100, min_dim),
        'rff_dim': 256,
        'n_landmarks': min(500, n_samples // 2),
        'use_spectral': True,
    }


def run_single_experiment(method, X_views, y_true, method_name):
    """Run a single clustering experiment."""
    start_time = time.time()
    
    try:
        if method_name == 'Best-View':
            labels = method.fit_predict(X_views)
        else:
            labels = method.fit_predict(X_views)
        
        elapsed = time.time() - start_time
        results = evaluate_clustering(y_true, labels)
        results['Time'] = elapsed
        
        return results
    except Exception as e:
        print(f"  Error in {method_name}: {e}")
        return None


def run_experiments(dataset_name, n_runs=10, save_dir='./results', random_seed=42,
                    include_deep=False, include_external=False, use_tuned=False):
    """
    Run experiments on a single dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    n_runs : int
        Number of runs for averaging.
    save_dir : str
        Directory to save results.
    random_seed : int
        Base random seed.
    include_deep : bool
        Whether to include scalable methods.
    include_external : bool
        Whether to include external methods (SCMVC, etc.).
    use_tuned : bool
        Whether to use Optuna-tuned parameters if available.

    Returns
    -------
    results_df : DataFrame
        Results table.
    """
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")

    # Load dataset
    dataset = load_dataset(dataset_name)
    dataset.normalize('standard')
    print(dataset)

    X_views = dataset.views
    y_true = dataset.labels
    n_clusters = dataset.n_clusters

    # Initialize methods (with optional scalable and external methods)
    methods = get_baseline_methods(n_clusters, random_state=random_seed,
                                   include_deep=include_deep, include_external=include_external)
    
    # Get SPOCK parameters (tuned or default)
    if use_tuned:
        tuned_params = load_tuned_params(dataset_name)
        if tuned_params:
            print(f"Using Optuna-tuned parameters for {dataset_name}")
            spock_params = tuned_params
        else:
            print(f"No tuned params found for {dataset_name}, using defaults")
            spock_params = get_default_spock_params(dataset, X_views)
    else:
        spock_params = get_default_spock_params(dataset, X_views)
    
    # Add SPOCK
    spock = SPOCK(
        n_clusters=n_clusters,
        random_state=random_seed,
        verbose=False,
        **spock_params
    )
    methods['SPOCK'] = spock
    
    # Track results
    trackers = {name: MetricTracker() for name in methods.keys()}
    
    # Run experiments
    for run in tqdm(range(n_runs), desc=f"Running {n_runs} experiments"):
        seed = random_seed + run
        np.random.seed(seed)
        
        for method_name, method in methods.items():
            # Update random state
            if hasattr(method, 'random_state'):
                method.random_state = seed
            
            results = run_single_experiment(method, X_views, y_true, method_name)
            
            if results is not None:
                trackers[method_name].add(results)
    
    # Compile results
    all_results = []
    metrics = ['ACC', 'NMI', 'ARI', 'Purity', 'F1']
    
    for method_name, tracker in trackers.items():
        stats = tracker.get_stats()
        row = {'Method': method_name}
        for metric in metrics:
            if metric in stats:
                row[f'{metric}_mean'] = stats[metric]['mean']
                row[f'{metric}_std'] = stats[metric]['std']
        all_results.append(row)
    
    results_df = pd.DataFrame(all_results)
    
    # Print summary
    print(f"\nResults Summary for {dataset_name}:")
    print("-" * 70)
    print(f"{'Method':<20}", end='')
    for metric in metrics:
        print(f"{metric:>12}", end='')
    print()
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        print(f"{row['Method']:<20}", end='')
        for metric in metrics:
            mean = row.get(f'{metric}_mean', 0)
            std = row.get(f'{metric}_std', 0)
            print(f"{mean:.3f}±{std:.3f}".rjust(12), end='')
        print()
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_path = os.path.join(save_dir, f'{dataset_name}_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    return results_df


def run_all_experiments(n_runs=10, save_dir='./results', random_seed=42,
                        include_deep=False, include_external=False, use_tuned=False):
    """Run experiments on all available datasets."""
    datasets = get_available_datasets()
    all_results = {}

    for dataset_name in datasets:
        try:
            results = run_experiments(
                dataset_name,
                n_runs=n_runs,
                save_dir=save_dir,
                random_seed=random_seed,
                include_deep=include_deep,
                include_external=include_external,
                use_tuned=use_tuned
            )
            all_results[dataset_name] = results
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue

    # Generate LaTeX table
    generate_latex_table(all_results, save_dir)

    return all_results


def generate_latex_table(all_results, save_dir):
    """Generate LaTeX table from results."""
    metrics = ['ACC', 'NMI', 'ARI', 'Purity', 'F1']
    
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Clustering performance comparison on multi-view datasets}")
    latex.append(r"\label{tab:results}")
    latex.append(r"\begin{tabular}{l" + "c" * len(metrics) + "}")
    latex.append(r"\toprule")
    latex.append("Method & " + " & ".join(metrics) + r" \\")
    latex.append(r"\midrule")
    
    for dataset_name, results_df in all_results.items():
        latex.append(r"\multicolumn{" + str(len(metrics) + 1) + r"}{c}{\textbf{" + dataset_name + r"}} \\")
        latex.append(r"\midrule")
        
        # Find best values
        best_values = {}
        for metric in metrics:
            col = f'{metric}_mean'
            if col in results_df.columns:
                best_values[metric] = results_df[col].max()
        
        for _, row in results_df.iterrows():
            cells = [row['Method']]
            for metric in metrics:
                mean = row.get(f'{metric}_mean', 0)
                std = row.get(f'{metric}_std', 0)
                
                if mean == best_values.get(metric, -1):
                    cells.append(f"\\textbf{{{mean:.2f}}}$\\pm${std:.2f}")
                else:
                    cells.append(f"{mean:.2f}$\\pm${std:.2f}")
            
            latex.append(" & ".join(cells) + r" \\")
        
        latex.append(r"\midrule")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")
    
    latex_path = os.path.join(save_dir, 'results_table.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"\nLaTeX table saved to: {latex_path}")


def main():
    parser = argparse.ArgumentParser(description='SPOCK Experiments')
    parser.add_argument('--dataset', type=str, default='all',
                       help='Dataset name or "all" for all datasets')
    parser.add_argument('--n_runs', type=int, default=10,
                       help='Number of runs for averaging')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--include_deep', '--include_scalable', action='store_true',
                       dest='include_scalable',
                       help='Include scalable SOTA baselines (LMVSC, SMVSC, BMVC, etc.)')
    parser.add_argument('--include_external', action='store_true',
                       help='Include external methods (SCMVC, etc.) that require git clone')
    parser.add_argument('--use-tuned', '--use_tuned', action='store_true',
                       dest='use_tuned',
                       help='Use Optuna-tuned parameters from config/tuned_params.json')

    args = parser.parse_args()

    # Print scalable methods availability if requested
    if args.include_scalable or args.include_external:
        from spock.baselines import check_scalable_methods_availability
        print("\nScalable Methods (near-linear complexity):")
        availability = check_scalable_methods_availability()
        for method, available in availability.items():
            status = "✓ Available" if available else "✗ Not available"
            print(f"  {method}: {status}")
        print()

    if args.dataset.lower() == 'all':
        run_all_experiments(
            n_runs=args.n_runs,
            save_dir=args.save_dir,
            random_seed=args.seed,
            include_deep=args.include_scalable,
            include_external=args.include_external,
            use_tuned=args.use_tuned
        )
    else:
        run_experiments(
            args.dataset,
            n_runs=args.n_runs,
            save_dir=args.save_dir,
            random_seed=args.seed,
            include_deep=args.include_scalable,
            include_external=args.include_external,
            use_tuned=args.use_tuned
        )


if __name__ == '__main__':
    main()
