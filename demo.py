"""
Quick Demo Script for SPOCK

A minimal example to verify the installation and basic functionality.

Usage:
    python demo.py
"""

import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spock import SPOCK
from spock.datasets import load_dataset
from spock.evaluation import evaluate_clustering, print_metrics
from spock.baselines import ConcatKMeans, MVSC


def main():
    print("="*60)
    print("SPOCK Demo")
    print("="*60)
    
    # Load synthetic dataset (will generate if real data not found)
    print("\n1. Loading Handwritten dataset (or synthetic equivalent)...")
    dataset = load_dataset('Handwritten')
    dataset.normalize('standard')
    print(f"   {dataset}")
    
    X_views = dataset.views
    y_true = dataset.labels
    n_clusters = dataset.n_clusters
    
    # Run SPOCK (Spectral mode)
    print("\n2. Running SPOCK (Spectral mode)...")
    spock_spectral = SPOCK(
        n_clusters=n_clusters,
        alpha=1.0,
        beta=0.1,
        lambda_l21=0.01,
        k_neighbors=10,
        proj_dim=min(50, min(v.shape[1] for v in X_views)),
        rff_dim=128,
        n_landmarks=min(300, dataset.n_samples // 2),
        max_iter=30,
        use_spectral=True,
        random_state=42,
        verbose=True
    )
    
    labels_spectral = spock_spectral.fit_predict(X_views)
    results_spectral = evaluate_clustering(y_true, labels_spectral)
    print_metrics(results_spectral, "SPOCK (Spectral)")
    
    # Run SPOCK (KMeans mode)
    print("\n2b. Running SPOCK (KMeans mode)...")
    spock_kmeans = SPOCK(
        n_clusters=n_clusters,
        alpha=1.0,
        beta=0.1,
        lambda_l21=0.01,
        k_neighbors=10,
        proj_dim=min(50, min(v.shape[1] for v in X_views)),
        use_spectral=False,
        random_state=42,
        verbose=True
    )
    
    labels_kmeans = spock_kmeans.fit_predict(X_views)
    results_kmeans = evaluate_clustering(y_true, labels_kmeans)
    print_metrics(results_kmeans, "SPOCK (KMeans)")
    
    # Compare with baseline
    print("\n3. Running baseline (Concat+KMeans)...")
    baseline = ConcatKMeans(n_clusters=n_clusters, random_state=42)
    labels_baseline = baseline.fit_predict(X_views)
    results_baseline = evaluate_clustering(y_true, labels_baseline)
    print_metrics(results_baseline, "Concat+KMeans")
    
    # Run MVSC
    print("\n4. Running MVSC...")
    mvsc = MVSC(n_clusters=n_clusters, random_state=42)
    labels_mvsc = mvsc.fit_predict(X_views)
    results_mvsc = evaluate_clustering(y_true, labels_mvsc)
    print_metrics(results_mvsc, "MVSC")
    
    # Summary comparison
    print("\n" + "="*60)
    print("Summary Comparison")
    print("="*60)
    print(f"{'Method':<20} {'ACC':<10} {'NMI':<10} {'ARI':<10}")
    print("-"*60)
    print(f"{'SPOCK (Spectral)':<20} {results_spectral['ACC']:<10.4f} {results_spectral['NMI']:<10.4f} {results_spectral['ARI']:<10.4f}")
    print(f"{'SPOCK (KMeans)':<20} {results_kmeans['ACC']:<10.4f} {results_kmeans['NMI']:<10.4f} {results_kmeans['ARI']:<10.4f}")
    print(f"{'Concat+KMeans':<20} {results_baseline['ACC']:<10.4f} {results_baseline['NMI']:<10.4f} {results_baseline['ARI']:<10.4f}")
    print(f"{'MVSC':<20} {results_mvsc['ACC']:<10.4f} {results_mvsc['NMI']:<10.4f} {results_mvsc['ARI']:<10.4f}")
    print("="*60)
    
    # Improvement (use best SPOCK result)
    best_spock_acc = max(results_spectral['ACC'], results_kmeans['ACC'])
    acc_improvement = (best_spock_acc - results_baseline['ACC']) / results_baseline['ACC'] * 100
    print(f"\nBest SPOCK improves ACC over Concat+KMeans by {acc_improvement:.1f}%")
    
    print("\nDemo completed successfully!")
    print("See experiments/ for comprehensive evaluation scripts.")


if __name__ == '__main__':
    main()
