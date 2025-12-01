"""
Scalability Analysis for SPOCK

Tests computational efficiency on varying data sizes.

Usage:
    python run_scalability.py --max_samples 50000
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spock import SPOCK
from spock.baselines import get_baseline_methods
from spock.evaluation import evaluate_clustering


def generate_scalability_data(n_samples, n_views=3, view_dims=[100, 150, 80], 
                               n_clusters=10, random_state=42):
    """Generate synthetic data for scalability testing."""
    np.random.seed(random_state)
    
    # Generate cluster labels
    labels = np.random.randint(0, n_clusters, n_samples)
    
    # Generate cluster centers
    latent_dim = 30
    centers = np.random.randn(n_clusters, latent_dim) * 5
    
    # Generate samples
    latent = np.zeros((n_samples, latent_dim))
    for k in range(n_clusters):
        mask = labels == k
        n_k = mask.sum()
        latent[mask] = centers[k] + np.random.randn(n_k, latent_dim)
    
    # Generate views
    views = []
    for dim in view_dims:
        projection = np.random.randn(latent_dim, dim) / np.sqrt(latent_dim)
        view = latent @ projection + np.random.randn(n_samples, dim) * 0.1
        views.append(view)
    
    return views, labels


def run_scalability_test(sample_sizes, n_runs=3, save_dir='./results/scalability',
                          random_seed=42, test_baselines=True):
    """
    Run scalability test.
    
    Parameters
    ----------
    sample_sizes : list
        List of sample sizes to test.
    n_runs : int
        Number of runs per size.
    save_dir : str
        Directory to save results.
    random_seed : int
        Random seed.
    test_baselines : bool
        Whether to also test baseline methods.
        
    Returns
    -------
    results_df : DataFrame
        Timing results.
    """
    print(f"\n{'='*70}")
    print("Scalability Analysis")
    print(f"Sample sizes: {sample_sizes}")
    print(f"{'='*70}")
    
    n_clusters = 10
    view_dims = [100, 150, 80]
    
    all_results = []
    
    for n_samples in tqdm(sample_sizes, desc="Testing sample sizes"):
        print(f"\nTesting n_samples = {n_samples}")
        
        # Generate data
        views, labels = generate_scalability_data(
            n_samples, n_views=3, view_dims=view_dims,
            n_clusters=n_clusters, random_state=random_seed
        )
        
        # Test SPOCK
        spock_times = []
        for run in range(n_runs):
            model = SPOCK(
                n_clusters=n_clusters,
                alpha=1.0,
                beta=0.1,
                lambda_l21=0.01,
                k_neighbors=10,
                proj_dim=50,
                rff_dim=256,
                n_landmarks=min(500, n_samples // 2),
                random_state=random_seed + run,
                verbose=False
            )
            
            start = time.time()
            _ = model.fit_predict(views)
            elapsed = time.time() - start
            spock_times.append(elapsed)
        
        all_results.append({
            'n_samples': n_samples,
            'method': 'SPOCK',
            'time_mean': np.mean(spock_times),
            'time_std': np.std(spock_times)
        })
        
        print(f"  SPOCK: {np.mean(spock_times):.2f}s ± {np.std(spock_times):.2f}s")
        
        # Test baselines (only for smaller sizes)
        if test_baselines and n_samples <= 10000:
            baselines = get_baseline_methods(n_clusters, random_state=random_seed)
            
            for method_name, method in baselines.items():
                try:
                    method_times = []
                    for run in range(n_runs):
                        if hasattr(method, 'random_state'):
                            method.random_state = random_seed + run
                        
                        start = time.time()
                        _ = method.fit_predict(views)
                        elapsed = time.time() - start
                        method_times.append(elapsed)
                    
                    all_results.append({
                        'n_samples': n_samples,
                        'method': method_name,
                        'time_mean': np.mean(method_times),
                        'time_std': np.std(method_times)
                    })
                    
                    print(f"  {method_name}: {np.mean(method_times):.2f}s ± {np.std(method_times):.2f}s")
                    
                except Exception as e:
                    print(f"  {method_name}: Failed ({e})")
    
    results_df = pd.DataFrame(all_results)
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_path = os.path.join(save_dir, f'scalability_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Generate visualization
    plot_scalability(results_df, save_dir)
    
    return results_df


def plot_scalability(results_df, save_dir):
    """Generate scalability plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = results_df['method'].unique()
    
    for method in methods:
        method_data = results_df[results_df['method'] == method]
        x = method_data['n_samples'].values
        y = method_data['time_mean'].values
        yerr = method_data['time_std'].values
        
        linestyle = '-' if method == 'SPOCK' else '--'
        linewidth = 2.5 if method == 'SPOCK' else 1.5
        
        ax.errorbar(x, y, yerr=yerr, marker='o', label=method,
                   linestyle=linestyle, linewidth=linewidth, capsize=3)
    
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Scalability Analysis', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    fig_path = os.path.join(save_dir, 'scalability.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {fig_path}")
    
    # Also plot theoretical complexity
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get SPOCK data
    spock_data = results_df[results_df['method'] == 'SPOCK']
    x = spock_data['n_samples'].values
    y = spock_data['time_mean'].values
    
    ax.plot(x, y, 'o-', label='SPOCK (Measured)', linewidth=2, markersize=8)
    
    # Theoretical O(N) line
    if len(x) > 0:
        scale = y[-1] / x[-1]
        ax.plot(x, scale * x, '--', label='O(N) Reference', alpha=0.7)
        
        # O(N^2) reference
        scale_n2 = y[0] / (x[0]**2)
        y_n2 = scale_n2 * np.array(x)**2
        ax.plot(x, y_n2, ':', label='O(N²) Reference', alpha=0.7)
    
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('SPOCK Scalability vs Theoretical Complexity', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    fig_path = os.path.join(save_dir, 'complexity_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Complexity analysis figure saved to: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description='SPOCK Scalability Analysis')
    parser.add_argument('--max_samples', type=int, default=50000,
                       help='Maximum number of samples to test')
    parser.add_argument('--n_runs', type=int, default=3,
                       help='Number of runs per size')
    parser.add_argument('--save_dir', type=str, default='./results/scalability',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no_baselines', action='store_true',
                       help='Skip baseline methods')
    
    args = parser.parse_args()
    
    # Generate sample sizes (log scale)
    sample_sizes = [500, 1000, 2000, 5000, 10000]
    if args.max_samples > 10000:
        sample_sizes.extend([20000, 50000])
    if args.max_samples > 50000:
        sample_sizes.append(args.max_samples)
    
    sample_sizes = [s for s in sample_sizes if s <= args.max_samples]
    
    run_scalability_test(
        sample_sizes,
        n_runs=args.n_runs,
        save_dir=args.save_dir,
        random_seed=args.seed,
        test_baselines=not args.no_baselines
    )


if __name__ == '__main__':
    main()
