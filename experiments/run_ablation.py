"""
Ablation Study Script for SPOCK

Analyzes the contribution of each component:
1. Feature Selection (Phase 1)
2. Density-Aware Graph (Phase 2)
3. OT Alignment (Phase 2)
4. Nyström Approximation (Phase 3)

Usage:
    python run_ablation.py --dataset Handwritten --n_runs 10
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spock import SPOCKAblation
from spock.datasets import load_dataset, get_available_datasets
from spock.evaluation import evaluate_clustering, MetricTracker


# Ablation configurations
ABLATION_CONFIGS = {
    'SPOCK (Full)': 'full',
    'w/o Feature Selection': 'no_feature_selection',
    'w/o Density-Aware': 'no_density_aware',
    'w/o OT Alignment': 'no_ot_alignment',
    'w/ Standard Spectral': 'standard_spectral',
}


def run_ablation_study(dataset_name, n_runs=10, save_dir='./results/ablation', 
                       random_seed=42):
    """
    Run ablation study on a single dataset.
    
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
        
    Returns
    -------
    results_df : DataFrame
        Ablation results.
    """
    print(f"\n{'='*70}")
    print(f"Ablation Study on: {dataset_name}")
    print(f"{'='*70}")
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    dataset.normalize('standard')
    print(dataset)
    
    X_views = dataset.views
    y_true = dataset.labels
    n_clusters = dataset.n_clusters
    
    # Track results
    trackers = {name: MetricTracker() for name in ABLATION_CONFIGS.keys()}
    
    # Run experiments
    for run in tqdm(range(n_runs), desc=f"Running {n_runs} ablation experiments"):
        seed = random_seed + run
        np.random.seed(seed)
        
        for config_name, ablation_mode in ABLATION_CONFIGS.items():
            # Create model with ablation
            model = SPOCKAblation(
                ablation_mode=ablation_mode,
                n_clusters=n_clusters,
                alpha=1.0,
                beta=0.1,
                lambda_l21=0.01,
                k_neighbors=10,
                proj_dim=min(100, min(v.shape[1] for v in X_views)),
                rff_dim=256,
                n_landmarks=min(500, dataset.n_samples // 2),
                random_state=seed,
                verbose=False
            )
            
            try:
                labels = model.fit_predict(X_views)
                results = evaluate_clustering(y_true, labels)
                trackers[config_name].add(results)
            except Exception as e:
                print(f"  Error in {config_name}: {e}")
    
    # Compile results
    all_results = []
    metrics = ['ACC', 'NMI', 'ARI', 'Purity', 'F1']
    
    for config_name, tracker in trackers.items():
        stats = tracker.get_stats()
        row = {'Configuration': config_name}
        for metric in metrics:
            if metric in stats:
                row[f'{metric}_mean'] = stats[metric]['mean']
                row[f'{metric}_std'] = stats[metric]['std']
        all_results.append(row)
    
    results_df = pd.DataFrame(all_results)
    
    # Print summary
    print(f"\nAblation Results for {dataset_name}:")
    print("-" * 80)
    print(f"{'Configuration':<25}", end='')
    for metric in metrics:
        print(f"{metric:>12}", end='')
    print()
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{row['Configuration']:<25}", end='')
        for metric in metrics:
            mean = row.get(f'{metric}_mean', 0)
            std = row.get(f'{metric}_std', 0)
            print(f"{mean:.3f}±{std:.3f}".rjust(12), end='')
        print()
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_path = os.path.join(save_dir, f'ablation_{dataset_name}_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Generate visualization
    plot_ablation_results(results_df, dataset_name, save_dir)
    
    return results_df


def plot_ablation_results(results_df, dataset_name, save_dir):
    """Generate bar chart for ablation results."""
    metrics = ['ACC', 'NMI', 'ARI', 'Purity', 'F1']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        configs = results_df['Configuration'].values
        means = results_df[f'{metric}_mean'].values
        stds = results_df[f'{metric}_std'].values
        
        colors = ['#2ecc71' if 'Full' in c else '#3498db' for c in configs]
        
        bars = ax.bar(range(len(configs)), means, yerr=stds, capsize=3, color=colors)
        
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([c.replace(' ', '\n') for c in configs], rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.0)
        
        # Highlight the full model
        for j, (bar, config) in enumerate(zip(bars, configs)):
            if 'Full' in config:
                bar.set_edgecolor('black')
                bar.set_linewidth(2)
    
    plt.suptitle(f'Ablation Study on {dataset_name}', fontsize=14)
    plt.tight_layout()
    
    fig_path = os.path.join(save_dir, f'ablation_{dataset_name}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {fig_path}")


def run_ablation_all_datasets(n_runs=10, save_dir='./results/ablation', random_seed=42):
    """Run ablation study on all datasets."""
    datasets = get_available_datasets()
    all_results = {}
    
    for dataset_name in datasets:
        try:
            results = run_ablation_study(
                dataset_name,
                n_runs=n_runs,
                save_dir=save_dir,
                random_seed=random_seed
            )
            all_results[dataset_name] = results
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
    
    # Generate combined table
    generate_ablation_latex_table(all_results, save_dir)
    
    return all_results


def generate_ablation_latex_table(all_results, save_dir):
    """Generate LaTeX table for ablation study."""
    metrics = ['ACC', 'NMI', 'ARI']
    
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Ablation study results}")
    latex.append(r"\label{tab:ablation}")
    latex.append(r"\resizebox{\linewidth}{!}{")
    
    n_datasets = len(all_results)
    latex.append(r"\begin{tabular}{l" + "ccc" * n_datasets + "}")
    latex.append(r"\toprule")
    
    # Header
    header = "Configuration"
    for dataset in all_results.keys():
        header += f" & \\multicolumn{{3}}{{c}}{{{dataset}}}"
    header += r" \\"
    latex.append(header)
    
    # Subheader
    subheader = ""
    for _ in all_results.keys():
        subheader += " & ACC & NMI & ARI"
    subheader += r" \\"
    latex.append(subheader)
    latex.append(r"\midrule")
    
    # Data rows
    for config in ABLATION_CONFIGS.keys():
        row = config
        for dataset_name, results_df in all_results.items():
            config_row = results_df[results_df['Configuration'] == config]
            if len(config_row) > 0:
                for metric in metrics:
                    mean = config_row[f'{metric}_mean'].values[0]
                    row += f" & {mean:.2f}"
            else:
                row += " & - & - & -"
        row += r" \\"
        latex.append(row)
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}}")
    latex.append(r"\end{table}")
    
    latex_path = os.path.join(save_dir, 'ablation_table.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"\nLaTeX table saved to: {latex_path}")


def main():
    parser = argparse.ArgumentParser(description='SPOCK Ablation Study')
    parser.add_argument('--dataset', type=str, default='all',
                       help='Dataset name or "all" for all datasets')
    parser.add_argument('--n_runs', type=int, default=10,
                       help='Number of runs for averaging')
    parser.add_argument('--save_dir', type=str, default='./results/ablation',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    if args.dataset.lower() == 'all':
        run_ablation_all_datasets(
            n_runs=args.n_runs,
            save_dir=args.save_dir,
            random_seed=args.seed
        )
    else:
        run_ablation_study(
            args.dataset,
            n_runs=args.n_runs,
            save_dir=args.save_dir,
            random_seed=args.seed
        )


if __name__ == '__main__':
    main()
