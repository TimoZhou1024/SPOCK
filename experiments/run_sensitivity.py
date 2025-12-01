"""
Parameter Sensitivity Analysis for SPOCK

Analyzes the impact of key hyperparameters:
- α (alpha): self-expression weight
- β (beta): S sparsity weight
- λ (lambda_l21): L2,1 regularization
- K (k_neighbors): number of neighbors
- RFF dimension
- Number of landmarks

Usage:
    python run_sensitivity.py --dataset Handwritten --param alpha
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spock import SPOCK
from spock.datasets import load_dataset
from spock.evaluation import evaluate_clustering, MetricTracker


# Parameter ranges
PARAM_RANGES = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'beta': [0.001, 0.01, 0.1, 1.0, 10.0],
    'lambda_l21': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'k_neighbors': [5, 10, 15, 20, 30, 50],
    'rff_dim': [64, 128, 256, 512, 1024],
    'n_landmarks': [100, 200, 300, 500, 800],
    'proj_dim': [20, 50, 100, 150, 200],
}


def run_sensitivity_analysis(dataset_name, param_name, n_runs=5, 
                              save_dir='./results/sensitivity', random_seed=42):
    """
    Run sensitivity analysis for a single parameter.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    param_name : str
        Name of the parameter to analyze.
    n_runs : int
        Number of runs per setting.
    save_dir : str
        Directory to save results.
    random_seed : int
        Base random seed.
        
    Returns
    -------
    results_df : DataFrame
        Sensitivity results.
    """
    print(f"\n{'='*70}")
    print(f"Sensitivity Analysis: {param_name} on {dataset_name}")
    print(f"{'='*70}")
    
    if param_name not in PARAM_RANGES:
        raise ValueError(f"Unknown parameter: {param_name}. "
                        f"Available: {list(PARAM_RANGES.keys())}")
    
    param_values = PARAM_RANGES[param_name]
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    dataset.normalize('standard')
    print(dataset)
    
    X_views = dataset.views
    y_true = dataset.labels
    n_clusters = dataset.n_clusters
    
    # Default parameters
    default_params = {
        'n_clusters': n_clusters,
        'alpha': 1.0,
        'beta': 0.1,
        'lambda_l21': 0.01,
        'k_neighbors': 10,
        'proj_dim': min(100, min(v.shape[1] for v in X_views)),
        'rff_dim': 256,
        'n_landmarks': min(500, dataset.n_samples // 2),
        'verbose': False
    }
    
    all_results = []
    
    for param_value in tqdm(param_values, desc=f"Testing {param_name}"):
        tracker = MetricTracker()
        
        for run in range(n_runs):
            seed = random_seed + run
            np.random.seed(seed)
            
            # Update parameter
            params = default_params.copy()
            params[param_name] = param_value
            params['random_state'] = seed
            
            # Ensure valid parameter
            if param_name == 'n_landmarks':
                params[param_name] = min(param_value, dataset.n_samples // 2)
            if param_name == 'proj_dim':
                params[param_name] = min(param_value, min(v.shape[1] for v in X_views))
            
            try:
                model = SPOCK(**params)
                labels = model.fit_predict(X_views)
                results = evaluate_clustering(y_true, labels)
                tracker.add(results)
            except Exception as e:
                print(f"  Error with {param_name}={param_value}: {e}")
        
        # Record statistics
        stats = tracker.get_stats()
        row = {'param_value': param_value}
        for metric in ['ACC', 'NMI', 'ARI', 'Purity', 'F1']:
            if metric in stats:
                row[f'{metric}_mean'] = stats[metric]['mean']
                row[f'{metric}_std'] = stats[metric]['std']
        all_results.append(row)
    
    results_df = pd.DataFrame(all_results)
    
    # Print summary
    print(f"\nSensitivity Results for {param_name}:")
    print("-" * 60)
    print(f"{'Value':<15} {'ACC':<12} {'NMI':<12} {'ARI':<12}")
    print("-" * 60)
    
    for _, row in results_df.iterrows():
        acc = row.get('ACC_mean', 0)
        nmi = row.get('NMI_mean', 0)
        ari = row.get('ARI_mean', 0)
        print(f"{row['param_value']:<15} {acc:.4f}       {nmi:.4f}       {ari:.4f}")
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_path = os.path.join(save_dir, f'sensitivity_{param_name}_{dataset_name}_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Generate visualization
    plot_sensitivity(results_df, param_name, dataset_name, save_dir)
    
    return results_df


def plot_sensitivity(results_df, param_name, dataset_name, save_dir):
    """Generate sensitivity analysis plot."""
    metrics = ['ACC', 'NMI', 'ARI']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = results_df['param_value'].values
    
    for metric in metrics:
        means = results_df[f'{metric}_mean'].values
        stds = results_df[f'{metric}_std'].values
        
        ax.errorbar(range(len(x)), means, yerr=stds, marker='o', 
                   label=metric, capsize=3)
    
    ax.set_xlabel(param_name)
    ax.set_ylabel('Score')
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([str(v) for v in x], rotation=45)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.set_title(f'Sensitivity Analysis: {param_name} on {dataset_name}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = os.path.join(save_dir, f'sensitivity_{param_name}_{dataset_name}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {fig_path}")


def run_all_sensitivity(dataset_name, n_runs=5, save_dir='./results/sensitivity', 
                         random_seed=42):
    """Run sensitivity analysis for all parameters."""
    all_results = {}
    
    for param_name in PARAM_RANGES.keys():
        try:
            results = run_sensitivity_analysis(
                dataset_name, param_name,
                n_runs=n_runs,
                save_dir=save_dir,
                random_seed=random_seed
            )
            all_results[param_name] = results
        except Exception as e:
            print(f"Error analyzing {param_name}: {e}")
            continue
    
    # Generate combined heatmap
    plot_sensitivity_heatmap(all_results, dataset_name, save_dir)
    
    return all_results


def plot_sensitivity_heatmap(all_results, dataset_name, save_dir):
    """Generate combined sensitivity heatmap."""
    params = list(all_results.keys())
    n_params = len(params)
    
    fig, axes = plt.subplots(1, n_params, figsize=(4*n_params, 4))
    if n_params == 1:
        axes = [axes]
    
    for i, param_name in enumerate(params):
        ax = axes[i]
        results_df = all_results[param_name]
        
        # Create heatmap data
        metrics = ['ACC', 'NMI', 'ARI']
        data = np.array([[results_df[f'{m}_mean'].values[j] 
                         for j in range(len(results_df))]
                        for m in metrics])
        
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(results_df)))
        ax.set_xticklabels([f"{v:.1e}" if v < 1 else str(v) 
                          for v in results_df['param_value']], 
                         rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metrics)
        ax.set_xlabel(param_name)
        ax.set_title(param_name)
        
        # Add color bar
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle(f'Parameter Sensitivity on {dataset_name}', fontsize=14)
    plt.tight_layout()
    
    fig_path = os.path.join(save_dir, f'sensitivity_heatmap_{dataset_name}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description='SPOCK Parameter Sensitivity Analysis')
    parser.add_argument('--dataset', type=str, default='Handwritten',
                       help='Dataset name')
    parser.add_argument('--param', type=str, default='all',
                       help='Parameter name or "all" for all parameters')
    parser.add_argument('--n_runs', type=int, default=5,
                       help='Number of runs per setting')
    parser.add_argument('--save_dir', type=str, default='./results/sensitivity',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    if args.param.lower() == 'all':
        run_all_sensitivity(
            args.dataset,
            n_runs=args.n_runs,
            save_dir=args.save_dir,
            random_seed=args.seed
        )
    else:
        run_sensitivity_analysis(
            args.dataset, args.param,
            n_runs=args.n_runs,
            save_dir=args.save_dir,
            random_seed=args.seed
        )


if __name__ == '__main__':
    main()
