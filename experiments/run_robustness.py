"""
Robustness Experiments for SPOCK

Tests SPOCK's robustness to:
1. Incomplete data (missing views)
2. Unaligned data (shuffled views)

Usage Examples:
    # Run missing view experiments
    python run_robustness.py --test missing --dataset Handwritten

    # Run unaligned view experiments  
    python run_robustness.py --test unaligned --dataset Handwritten

    # Run both tests on all datasets
    python run_robustness.py --test all --dataset all

    # Quick test with fewer runs
    python run_robustness.py --test all --dataset Handwritten --n_runs 3
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spock import SPOCK
from spock.datasets import load_dataset, get_available_datasets
from spock.baselines import get_baseline_methods
from spock.evaluation import evaluate_clustering


# Path configurations
TUNED_PARAMS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'config', 'tuned_params.json'
)
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results', 'robustness'
)


def load_tuned_params(dataset_name: str) -> dict:
    """Load Optuna-tuned parameters for a dataset."""
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
    """Get default SPOCK parameters based on dataset characteristics."""
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


def create_missing_views(views, missing_rate, random_state=None):
    """
    Create incomplete multi-view data by setting missing views to NaN or zeros.
    
    Parameters
    ----------
    views : list of ndarray
        Original views
    missing_rate : float
        Fraction of samples to have at least one missing view
    random_state : int, optional
        Random seed
    
    Returns
    -------
    incomplete_views : list of ndarray
        Views with some entries set to indicate missing (zeros with mask)
    mask : ndarray of shape (n_samples, n_views)
        Boolean mask where True = view is available
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = views[0].shape[0]
    n_views = len(views)
    
    # Create mask: True = available
    mask = np.ones((n_samples, n_views), dtype=bool)
    
    # Determine which samples will have missing views
    n_missing_samples = int(missing_rate * n_samples)
    missing_sample_idx = np.random.choice(n_samples, n_missing_samples, replace=False)
    
    for idx in missing_sample_idx:
        # Each missing sample loses 1 to (n_views-1) views (keep at least one)
        n_remove = np.random.randint(1, n_views)
        remove_views = np.random.choice(n_views, n_remove, replace=False)
        mask[idx, remove_views] = False
    
    # Create copies with missing data (set to zeros where missing)
    incomplete_views = []
    for v_idx, view in enumerate(views):
        new_view = view.copy()
        missing_mask = ~mask[:, v_idx]
        new_view[missing_mask] = 0  # Set missing entries to zero
        incomplete_views.append(new_view)
    
    return incomplete_views, mask


def create_unaligned_views(views, shuffle_rate, random_state=None):
    """
    Create unaligned multi-view data by shuffling samples in some views.
    
    Parameters
    ----------
    views : list of ndarray
        Original views (assumes first view is kept aligned)
    shuffle_rate : float
        Fraction of samples to shuffle in non-first views
    random_state : int, optional
        Random seed
    
    Returns
    -------
    unaligned_views : list of ndarray
        Views with shuffled samples (first view unchanged)
    shuffle_info : dict
        Information about the shuffling applied
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = views[0].shape[0]
    n_views = len(views)
    
    # First view stays aligned
    unaligned_views = [views[0].copy()]
    
    shuffle_info = {'shuffle_rate': shuffle_rate, 'view_shuffles': {}}
    
    # Shuffle other views
    for v_idx in range(1, n_views):
        new_view = views[v_idx].copy()
        
        if shuffle_rate > 0:
            # Select samples to shuffle
            n_shuffle = int(shuffle_rate * n_samples)
            shuffle_idx = np.random.choice(n_samples, n_shuffle, replace=False)
            
            # Create random permutation for selected samples
            shuffled_positions = np.random.permutation(shuffle_idx)
            
            # Apply shuffle
            new_view[shuffle_idx] = views[v_idx][shuffled_positions]
            
            shuffle_info['view_shuffles'][v_idx] = {
                'original_idx': shuffle_idx.tolist(),
                'new_idx': shuffled_positions.tolist()
            }
        
        unaligned_views.append(new_view)
    
    return unaligned_views, shuffle_info


def impute_missing_views(views, mask, method='mean'):
    """
    Simple imputation for missing views.
    
    Parameters
    ----------
    views : list of ndarray
        Views with missing data (zeros where missing)
    mask : ndarray
        Boolean mask where True = available
    method : str
        'mean' - replace with view mean
        'zero' - keep zeros (already done)
        'neighbor' - use nearest neighbor from same view
    
    Returns
    -------
    imputed_views : list of ndarray
    """
    if method == 'zero':
        return views
    
    imputed_views = []
    for v_idx, view in enumerate(views):
        new_view = view.copy()
        missing_mask = ~mask[:, v_idx]
        
        if method == 'mean':
            # Compute mean from available samples
            available_mask = mask[:, v_idx]
            if available_mask.sum() > 0:
                view_mean = view[available_mask].mean(axis=0)
                new_view[missing_mask] = view_mean
        
        imputed_views.append(new_view)
    
    return imputed_views


def get_comparison_methods(n_clusters, include_external=True):
    """
    Get comparison methods for robustness testing.
    
    Returns external methods (SCMVC, EFIMVC, ALPC, RCAGL, ROLL) for comparison.
    """
    methods = {}
    
    if include_external:
        try:
            from spock.baselines import SCMVCWrapper
            methods['SCMVC'] = SCMVCWrapper(n_clusters=n_clusters, pre_epochs=50, con_epochs=20)
        except Exception as e:
            print(f"Warning: Could not load SCMVC: {e}")
        
        try:
            from spock.baselines import EFIMVCWrapper
            methods['EFIMVC'] = EFIMVCWrapper(n_clusters=n_clusters)
        except Exception as e:
            print(f"Warning: Could not load EFIMVC: {e}")
        
        try:
            from spock.baselines import ALPCWrapper
            methods['ALPC'] = ALPCWrapper(n_clusters=n_clusters)
        except Exception as e:
            print(f"Warning: Could not load ALPC: {e}")
        
        try:
            from spock.baselines import RCAGLWrapper
            methods['RCAGL'] = RCAGLWrapper(n_clusters=n_clusters)
        except Exception as e:
            print(f"Warning: Could not load RCAGL: {e}")
        
        try:
            from spock.baselines import ROLLWrapper
            methods['ROLL'] = ROLLWrapper(n_clusters=n_clusters, warm_epochs=20, epochs=50)
        except Exception as e:
            print(f"Warning: Could not load ROLL: {e}")
    
    return methods


def run_single_experiment(model, views, true_labels, model_name):
    """Run a single clustering experiment and return results."""
    try:
        start_time = time.time()
        pred_labels = model.fit_predict(views)
        elapsed = time.time() - start_time
        
        metrics = evaluate_clustering(true_labels, pred_labels)
        return {
            'ACC': metrics['ACC'],
            'NMI': metrics['NMI'],
            'Purity': metrics['Purity'],
            'ARI': metrics['ARI'],
            'time': elapsed,
            'success': True
        }
    except Exception as e:
        print(f"  Warning: {model_name} failed: {e}")
        return {
            'ACC': 0.0, 'NMI': 0.0, 'Purity': 0.0, 'ARI': 0.0,
            'time': 0.0, 'success': False
        }


def run_missing_view_experiment(dataset_name, missing_rates, n_runs=5, 
                                 include_external=True, use_tuned=True, verbose=True):
    """
    Run incomplete data robustness experiment.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name
    missing_rates : list of float
        Missing rates to test (e.g., [0.0, 0.1, 0.3, 0.5, 0.7])
    n_runs : int
        Number of runs per setting
    include_external : bool
        Whether to include external methods (SCMVC, EFIMVC, ALPC)
    use_tuned : bool
        Whether to use tuned SPOCK parameters
    verbose : bool
        Print progress
    
    Returns
    -------
    results_df : DataFrame
        Results with columns: method, missing_rate, metric, value, run
    """
    # Load dataset
    dataset = load_dataset(dataset_name)
    dataset.normalize('standard')
    X_views = dataset.views
    true_labels = dataset.labels
    n_clusters = dataset.n_clusters
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Missing View Robustness Test: {dataset_name}")
        print(f"  Samples: {dataset.n_samples}, Views: {dataset.n_views}, Clusters: {n_clusters}")
        print(f"  Missing rates: {missing_rates}")
        print(f"  Runs per setting: {n_runs}")
        print(f"{'='*60}")
    
    # Get SPOCK parameters
    tuned_params = load_tuned_params(dataset_name) if use_tuned else None
    base_params = tuned_params if tuned_params else get_default_spock_params(dataset, X_views)
    
    # Prepare methods
    methods = get_comparison_methods(n_clusters, include_external)
    
    results = []
    
    for missing_rate in missing_rates:
        if verbose:
            print(f"\n--- Missing Rate: {missing_rate:.0%} ---")
        
        for run in range(n_runs):
            seed = 42 + run
            
            # Create incomplete data
            incomplete_views, mask = create_missing_views(
                X_views, missing_rate, random_state=seed
            )
            
            # Simple imputation for baselines
            imputed_views = impute_missing_views(incomplete_views, mask, method='mean')
            
            # Run SPOCK (uses its own handling via OT alignment)
            spock_params = base_params.copy()
            spock_params['n_clusters'] = n_clusters
            spock_params['random_state'] = seed
            spock_params['verbose'] = False
            
            spock = SPOCK(**spock_params)
            spock_result = run_single_experiment(
                spock, imputed_views, true_labels, 'SPOCK'
            )
            
            for metric in ['ACC', 'NMI', 'Purity', 'ARI']:
                results.append({
                    'method': 'SPOCK',
                    'missing_rate': missing_rate,
                    'metric': metric,
                    'value': spock_result[metric],
                    'run': run,
                    'success': spock_result['success']
                })
            
            # Run external methods
            for method_name, method in methods.items():
                # Reset method state
                if hasattr(method, 'random_state'):
                    method.random_state = seed
                
                result = run_single_experiment(
                    method, imputed_views, true_labels, method_name
                )
                
                for metric in ['ACC', 'NMI', 'Purity', 'ARI']:
                    results.append({
                        'method': method_name,
                        'missing_rate': missing_rate,
                        'metric': metric,
                        'value': result[metric],
                        'run': run,
                        'success': result['success']
                    })
            
            if verbose and run == 0:
                print(f"  SPOCK ACC: {spock_result['ACC']:.4f}")
    
    return pd.DataFrame(results)


def run_unaligned_view_experiment(dataset_name, shuffle_rates, n_runs=5,
                                   include_external=True, use_tuned=True,
                                   verbose=True):
    """
    Run unaligned data robustness experiment.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name
    shuffle_rates : list of float
        Shuffle rates to test (e.g., [0.0, 0.2, 0.4, 0.6])
    n_runs : int
        Number of runs per setting
    include_external : bool
        Whether to include external methods (SCMVC, EFIMVC, ALPC)
    use_tuned : bool
        Whether to use tuned SPOCK parameters
    verbose : bool
        Print progress
    
    Returns
    -------
    results_df : DataFrame
        Results with columns: method, shuffle_rate, metric, value, run
    """
    # Load dataset
    dataset = load_dataset(dataset_name)
    dataset.normalize('standard')
    X_views = dataset.views
    true_labels = dataset.labels
    n_clusters = dataset.n_clusters
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Unaligned View Robustness Test: {dataset_name}")
        print(f"  Samples: {dataset.n_samples}, Views: {dataset.n_views}, Clusters: {n_clusters}")
        print(f"  Shuffle rates: {shuffle_rates}")
        print(f"  Runs per setting: {n_runs}")
        print(f"{'='*60}")
    
    # Get SPOCK parameters
    tuned_params = load_tuned_params(dataset_name) if use_tuned else None
    base_params = tuned_params if tuned_params else get_default_spock_params(dataset, X_views)
    
    # Prepare methods
    methods = get_comparison_methods(n_clusters, include_external)
    
    results = []
    
    for shuffle_rate in shuffle_rates:
        if verbose:
            print(f"\n--- Shuffle Rate: {shuffle_rate:.0%} ---")
        
        for run in range(n_runs):
            seed = 42 + run
            
            # Create unaligned data
            unaligned_views, shuffle_info = create_unaligned_views(
                X_views, shuffle_rate, random_state=seed
            )
            
            # Run SPOCK (uses OT for alignment)
            spock_params = base_params.copy()
            spock_params['n_clusters'] = n_clusters
            spock_params['random_state'] = seed
            spock_params['verbose'] = False
            
            spock = SPOCK(**spock_params)
            spock_result = run_single_experiment(
                spock, unaligned_views, true_labels, 'SPOCK'
            )
            
            for metric in ['ACC', 'NMI', 'Purity', 'ARI']:
                results.append({
                    'method': 'SPOCK',
                    'shuffle_rate': shuffle_rate,
                    'metric': metric,
                    'value': spock_result[metric],
                    'run': run,
                    'success': spock_result['success']
                })
            
            # Run external methods (they don't handle unalignment)
            for method_name, method in methods.items():
                if hasattr(method, 'random_state'):
                    method.random_state = seed
                
                result = run_single_experiment(
                    method, unaligned_views, true_labels, method_name
                )
                
                for metric in ['ACC', 'NMI', 'Purity', 'ARI']:
                    results.append({
                        'method': method_name,
                        'shuffle_rate': shuffle_rate,
                        'metric': metric,
                        'value': result[metric],
                        'run': run,
                        'success': result['success']
                    })
            
            if verbose and run == 0:
                print(f"  SPOCK ACC: {spock_result['ACC']:.4f}")
    
    return pd.DataFrame(results)


def plot_robustness_results(results_df, test_type, dataset_name, save_path=None):
    """
    Plot robustness experiment results.
    
    Parameters
    ----------
    results_df : DataFrame
        Results from robustness experiment
    test_type : str
        'missing' or 'unaligned'
    dataset_name : str
        Dataset name for title
    save_path : str, optional
        Path to save figure
    """
    rate_col = 'missing_rate' if test_type == 'missing' else 'shuffle_rate'
    rate_label = 'Missing Rate' if test_type == 'missing' else 'Shuffle Rate'
    title_prefix = 'Incomplete Data' if test_type == 'missing' else 'Unaligned Data'
    
    # Aggregate results: mean and std per method/rate/metric
    agg_results = results_df.groupby(['method', rate_col, 'metric'])['value'].agg(
        ['mean', 'std']
    ).reset_index()
    
    methods = results_df['method'].unique()
    rates = sorted(results_df[rate_col].unique())
    
    # Plot ACC and NMI
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for metric_idx, metric in enumerate(['ACC', 'NMI']):
        ax = axes[metric_idx]
        
        for m_idx, method in enumerate(methods):
            method_data = agg_results[
                (agg_results['method'] == method) & 
                (agg_results['metric'] == metric)
            ]
            
            means = method_data['mean'].values
            stds = method_data['std'].values
            rate_values = method_data[rate_col].values
            
            # Sort by rate
            sort_idx = np.argsort(rate_values)
            rate_values = rate_values[sort_idx]
            means = means[sort_idx]
            stds = stds[sort_idx]
            
            # Line style: SPOCK is bold
            linewidth = 3 if method == 'SPOCK' else 1.5
            linestyle = '-' if method == 'SPOCK' else '--'
            
            ax.errorbar(
                rate_values, means, yerr=stds,
                label=method, color=colors[m_idx],
                marker=markers[m_idx % len(markers)],
                linewidth=linewidth, linestyle=linestyle,
                capsize=3, markersize=8
            )
        
        ax.set_xlabel(rate_label, fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} vs {rate_label}', fontsize=14)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    
    fig.suptitle(f'{title_prefix} Robustness: {dataset_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.close()
    return fig


def print_summary_table(results_df, test_type):
    """Print a summary table of results."""
    rate_col = 'missing_rate' if test_type == 'missing' else 'shuffle_rate'
    
    # Pivot to show ACC for each method at each rate
    pivot = results_df[results_df['metric'] == 'ACC'].pivot_table(
        index='method', columns=rate_col, values='value',
        aggfunc='mean'
    )
    
    print(f"\n{'='*60}")
    print(f"Summary: ACC by {rate_col}")
    print(f"{'='*60}")
    print(pivot.round(4).to_string())
    
    # Compute degradation (drop from rate=0)
    if 0.0 in pivot.columns:
        print(f"\n--- Performance Drop from Baseline (rate=0) ---")
        baseline = pivot[0.0]
        for col in pivot.columns:
            if col > 0:
                drop = baseline - pivot[col]
                print(f"\nAt rate={col:.0%}:")
                for method in pivot.index:
                    print(f"  {method}: -{drop[method]:.4f} ({drop[method]/baseline[method]*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='SPOCK Robustness Experiments')
    parser.add_argument('--test', type=str, default='all',
                        choices=['missing', 'unaligned', 'all'],
                        help='Which robustness test to run')
    parser.add_argument('--dataset', type=str, default='Handwritten',
                        help='Dataset name or "all" for all datasets')
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Number of runs per setting')
    parser.add_argument('--no_external', action='store_true',
                        help='Run SPOCK only without external methods')
    parser.add_argument('--use_tuned', action='store_true', default=True,
                        help='Use Optuna-tuned parameters')
    parser.add_argument('--no_tuned', action='store_true',
                        help='Use default parameters instead of tuned')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save results and figures')
    
    args = parser.parse_args()
    
    use_tuned = args.use_tuned and not args.no_tuned
    include_external = not args.no_external
    
    # Datasets to test
    if args.dataset.lower() == 'all':
        datasets = ['Handwritten', 'Caltech101-7', 'BBCSport']
    else:
        datasets = [args.dataset]
    
    # Test configurations
    missing_rates = [0.0, 0.1, 0.3, 0.5, 0.7]
    shuffle_rates = [0.0, 0.2, 0.4, 0.6]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for dataset_name in datasets:
        print(f"\n{'#'*70}")
        print(f"# Dataset: {dataset_name}")
        print(f"{'#'*70}")
        
        # Missing view experiment
        if args.test in ['missing', 'all']:
            missing_results = run_missing_view_experiment(
                dataset_name, missing_rates, n_runs=args.n_runs,
                include_external=include_external, use_tuned=use_tuned
            )
            
            print_summary_table(missing_results, 'missing')
            
            if args.save:
                # Save CSV
                csv_path = os.path.join(
                    RESULTS_DIR, 
                    f'{dataset_name}_missing_{timestamp}.csv'
                )
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                missing_results.to_csv(csv_path, index=False)
                print(f"Results saved to: {csv_path}")
                
                # Save plot
                fig_path = os.path.join(
                    RESULTS_DIR,
                    f'{dataset_name}_missing_{timestamp}.png'
                )
                plot_robustness_results(
                    missing_results, 'missing', dataset_name, fig_path
                )
        
        # Unaligned view experiment
        if args.test in ['unaligned', 'all']:
            unaligned_results = run_unaligned_view_experiment(
                dataset_name, shuffle_rates, n_runs=args.n_runs,
                include_external=include_external, use_tuned=use_tuned
            )
            
            print_summary_table(unaligned_results, 'unaligned')
            
            if args.save:
                # Save CSV
                csv_path = os.path.join(
                    RESULTS_DIR,
                    f'{dataset_name}_unaligned_{timestamp}.csv'
                )
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                unaligned_results.to_csv(csv_path, index=False)
                print(f"Results saved to: {csv_path}")
                
                # Save plot
                fig_path = os.path.join(
                    RESULTS_DIR,
                    f'{dataset_name}_unaligned_{timestamp}.png'
                )
                plot_robustness_results(
                    unaligned_results, 'unaligned', dataset_name, fig_path
                )
    
    print(f"\n{'='*70}")
    print("Robustness experiments completed!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
