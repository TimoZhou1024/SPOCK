"""
Main Experiment Script for SPOCK

Compares SPOCK with baseline methods on multiple datasets.

Usage Examples:
    # SPOCK only (no baselines)
    python run_experiments.py --dataset Handwritten --spock_only

    # SPOCK + traditional baselines (default)
    python run_experiments.py --dataset Handwritten

    # SPOCK + scalable SOTA methods (LMVSC, SMVSC, etc.)
    python run_experiments.py --dataset Handwritten --include_scalable

    # SPOCK + external methods (SCMVC, etc.)
    python run_experiments.py --dataset Handwritten --include_external

    # All methods (traditional + scalable + external)
    python run_experiments.py --dataset Handwritten --include_all

    # Use Optuna-tuned parameters
    python run_experiments.py --dataset Handwritten --use_tuned
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
        labels = method.fit_predict(X_views)

        elapsed = time.time() - start_time
        results = evaluate_clustering(y_true, labels)
        results['Time'] = elapsed

        return results
    except Exception as e:
        print(f"  Error in {method_name}: {e}")
        return None


def run_experiments(dataset_name, n_runs=10, save_dir='./results', random_seed=42,
                    spock_only=False, include_traditional=True,
                    include_scalable=False, include_external=False, use_tuned=False):
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
    spock_only : bool
        If True, only run SPOCK (no baselines).
    include_traditional : bool
        Whether to include traditional baseline methods.
    include_scalable : bool
        Whether to include scalable SOTA methods (LMVSC, SMVSC, etc.).
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

    # Initialize methods based on flags
    methods = {}

    if not spock_only:
        if include_traditional:
            # Get traditional baseline methods
            traditional_methods = get_baseline_methods(
                n_clusters,
                random_state=random_seed,
                include_deep=False,
                include_external=False
            )
            methods.update(traditional_methods)

        if include_scalable:
            # Get scalable SOTA methods
            scalable_methods = get_baseline_methods(
                n_clusters,
                random_state=random_seed,
                include_deep=True,
                include_external=False
            )
            # Only add scalable methods (exclude traditional if already added)
            for name, method in scalable_methods.items():
                if name not in methods:
                    methods[name] = method

        if include_external:
            # Get external methods
            external_methods = get_baseline_methods(
                n_clusters,
                random_state=random_seed,
                include_deep=True,
                include_external=True
            )
            # Only add external methods
            for name, method in external_methods.items():
                if name not in methods:
                    methods[name] = method

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

    # Add SPOCK (always included)
    spock = SPOCK(
        n_clusters=n_clusters,
        random_state=random_seed,
        verbose=False,
        **spock_params
    )
    methods['SPOCK'] = spock

    # Print method summary
    print(f"\nMethods to run ({len(methods)}):")
    for name in methods.keys():
        print(f"  - {name}")
    print()

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
                        spock_only=False, include_traditional=True,
                        include_scalable=False, include_external=False, use_tuned=False):
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
                spock_only=spock_only,
                include_traditional=include_traditional,
                include_scalable=include_scalable,
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
    parser = argparse.ArgumentParser(
        description='SPOCK Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SPOCK only (no baselines)
  python run_experiments.py --dataset Handwritten --spock_only

  # SPOCK + traditional baselines (default)
  python run_experiments.py --dataset Handwritten

  # SPOCK + scalable SOTA methods
  python run_experiments.py --dataset Handwritten --include_scalable

  # SPOCK + external methods (SCMVC)
  python run_experiments.py --dataset Handwritten --include_external

  # All methods combined
  python run_experiments.py --dataset Handwritten --include_all

  # Scalable + external only (no traditional)
  python run_experiments.py --dataset Handwritten --include_scalable --include_external --no_traditional
        """
    )

    # Dataset and basic options
    parser.add_argument('--dataset', type=str, default='all',
                       help='Dataset name or "all" for all datasets')
    parser.add_argument('--n_runs', type=int, default=10,
                       help='Number of runs for averaging')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Method selection flags
    method_group = parser.add_argument_group('Method Selection')
    method_group.add_argument('--spock_only', action='store_true',
                             help='Run SPOCK only (no baseline methods)')
    method_group.add_argument('--no_traditional', action='store_true',
                             help='Exclude traditional baseline methods')
    method_group.add_argument('--include_scalable', '--include_deep', action='store_true',
                             dest='include_scalable',
                             help='Include scalable SOTA methods (LMVSC, SMVSC, BMVC, etc.)')
    method_group.add_argument('--include_external', action='store_true',
                             help='Include external methods (SCMVC, etc.) requiring git clone')
    method_group.add_argument('--include_all', action='store_true',
                             help='Include all methods (traditional + scalable + external)')

    # SPOCK parameters
    parser.add_argument('--use_tuned', '--use-tuned', action='store_true',
                       dest='use_tuned',
                       help='Use Optuna-tuned parameters from config/tuned_params.json')

    args = parser.parse_args()

    # Process flags
    spock_only = args.spock_only
    include_traditional = not args.no_traditional and not args.spock_only
    include_scalable = args.include_scalable or args.include_all
    include_external = args.include_external or args.include_all

    # If include_all, also include traditional
    if args.include_all:
        include_traditional = True

    # Print configuration
    print("\n" + "="*70)
    print("SPOCK Experiment Configuration")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Number of runs: {args.n_runs}")
    print(f"Random seed: {args.seed}")
    print(f"Use tuned params: {args.use_tuned}")
    print()
    print("Method Selection:")
    print(f"  SPOCK only: {spock_only}")
    print(f"  Traditional baselines: {include_traditional}")
    print(f"  Scalable SOTA methods: {include_scalable}")
    print(f"  External methods: {include_external}")

    # Print scalable methods availability if needed
    if include_scalable or include_external:
        from spock.baselines import check_scalable_methods_availability
        print("\nScalable/External Methods Availability:")
        availability = check_scalable_methods_availability()
        for method, available in availability.items():
            status = "Available" if available else "Not available"
            print(f"  {method}: {status}")

    # Run experiments
    if args.dataset.lower() == 'all':
        run_all_experiments(
            n_runs=args.n_runs,
            save_dir=args.save_dir,
            random_seed=args.seed,
            spock_only=spock_only,
            include_traditional=include_traditional,
            include_scalable=include_scalable,
            include_external=include_external,
            use_tuned=args.use_tuned
        )
    else:
        run_experiments(
            args.dataset,
            n_runs=args.n_runs,
            save_dir=args.save_dir,
            random_seed=args.seed,
            spock_only=spock_only,
            include_traditional=include_traditional,
            include_scalable=include_scalable,
            include_external=include_external,
            use_tuned=args.use_tuned
        )


if __name__ == '__main__':
    main()
