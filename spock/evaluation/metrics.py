"""
Clustering Evaluation Metrics

Includes: Accuracy (ACC), Normalized Mutual Information (NMI), 
Adjusted Rand Index (ARI), Purity, F-score, Precision, Recall
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    f1_score,
    precision_score,
    recall_score
)
from collections import Counter


def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy using Hungarian algorithm.
    
    ACC = max_mapping sum(I(y_true[i] == mapping(y_pred[i]))) / N
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted cluster labels.
        
    Returns
    -------
    acc : float
        Clustering accuracy (0-1).
    """
    y_true = np.array(y_true).astype(np.int64)
    y_pred = np.array(y_pred).astype(np.int64)
    
    assert y_true.shape[0] == y_pred.shape[0], "Label arrays must have same length"
    
    # Build cost matrix
    n_samples = y_true.shape[0]
    n_classes = max(y_true.max(), y_pred.max()) + 1
    
    # Count co-occurrence matrix
    cost_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(n_samples):
        cost_matrix[y_pred[i], y_true[i]] += 1
    
    # Hungarian algorithm (maximize matching -> minimize negative)
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    
    # Calculate accuracy
    correct = cost_matrix[row_ind, col_ind].sum()
    acc = correct / n_samples
    
    return acc


def clustering_nmi(y_true, y_pred):
    """
    Normalized Mutual Information.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like  
        Predicted cluster labels.
        
    Returns
    -------
    nmi : float
        NMI score (0-1).
    """
    return normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')


def clustering_ari(y_true, y_pred):
    """
    Adjusted Rand Index.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted cluster labels.
        
    Returns
    -------
    ari : float
        ARI score (-1 to 1, 1 is perfect).
    """
    return adjusted_rand_score(y_true, y_pred)


def clustering_purity(y_true, y_pred):
    """
    Clustering Purity.
    
    Purity = (1/N) * sum_k max_j |C_k ∩ T_j|
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted cluster labels.
        
    Returns
    -------
    purity : float
        Purity score (0-1).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    n_samples = y_true.shape[0]
    clusters = np.unique(y_pred)
    
    total_correct = 0
    for cluster in clusters:
        mask = y_pred == cluster
        if mask.sum() > 0:
            # Find most common true label in this cluster
            counter = Counter(y_true[mask])
            most_common = counter.most_common(1)[0][1]
            total_correct += most_common
    
    purity = total_correct / n_samples
    return purity


def clustering_fscore(y_true, y_pred, average='macro'):
    """
    F-score for clustering (after Hungarian matching).
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted cluster labels.
    average : str
        'macro', 'micro', or 'weighted'.
        
    Returns
    -------
    fscore : float
        F-score.
    """
    # Map clusters to true labels using Hungarian
    y_pred_mapped = map_clusters_to_labels(y_true, y_pred)
    return f1_score(y_true, y_pred_mapped, average=average, zero_division=0)


def clustering_precision(y_true, y_pred, average='macro'):
    """
    Precision for clustering (after Hungarian matching).
    """
    y_pred_mapped = map_clusters_to_labels(y_true, y_pred)
    return precision_score(y_true, y_pred_mapped, average=average, zero_division=0)


def clustering_recall(y_true, y_pred, average='macro'):
    """
    Recall for clustering (after Hungarian matching).
    """
    y_pred_mapped = map_clusters_to_labels(y_true, y_pred)
    return recall_score(y_true, y_pred_mapped, average=average, zero_division=0)


def map_clusters_to_labels(y_true, y_pred):
    """
    Map cluster assignments to true labels using Hungarian algorithm.
    
    Returns predicted labels mapped to true label space.
    """
    y_true = np.array(y_true).astype(np.int64)
    y_pred = np.array(y_pred).astype(np.int64)
    
    n_samples = y_true.shape[0]
    n_classes = max(y_true.max(), y_pred.max()) + 1
    
    # Build cost matrix
    cost_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(n_samples):
        cost_matrix[y_pred[i], y_true[i]] += 1
    
    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    
    # Create mapping
    mapping = dict(zip(row_ind, col_ind))
    
    # Map predictions
    y_pred_mapped = np.array([mapping.get(p, p) for p in y_pred])
    
    return y_pred_mapped


def evaluate_clustering(y_true, y_pred, metrics=None):
    """
    Comprehensive clustering evaluation.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted cluster labels.
    metrics : list, optional
        List of metrics to compute. If None, compute all.
        Options: 'ACC', 'NMI', 'ARI', 'Purity', 'F1', 'Precision', 'Recall'
        
    Returns
    -------
    results : dict
        Dictionary of metric names and values.
    """
    if metrics is None:
        metrics = ['ACC', 'NMI', 'ARI', 'Purity', 'F1', 'Precision', 'Recall']
    
    results = {}
    
    metric_funcs = {
        'ACC': clustering_accuracy,
        'NMI': clustering_nmi,
        'ARI': clustering_ari,
        'Purity': clustering_purity,
        'F1': lambda y, p: clustering_fscore(y, p, 'macro'),
        'Precision': lambda y, p: clustering_precision(y, p, 'macro'),
        'Recall': lambda y, p: clustering_recall(y, p, 'macro'),
    }
    
    for metric in metrics:
        if metric in metric_funcs:
            results[metric] = metric_funcs[metric](y_true, y_pred)
    
    return results


def print_metrics(results, dataset_name='Dataset'):
    """
    Print evaluation metrics in a formatted table.
    
    Parameters
    ----------
    results : dict
        Dictionary from evaluate_clustering.
    dataset_name : str
        Name of dataset for header.
    """
    print(f"\n{'='*60}")
    print(f"Clustering Results on {dataset_name}")
    print(f"{'='*60}")
    
    for metric, value in results.items():
        print(f"  {metric:12s}: {value:.4f}")
    
    print(f"{'='*60}\n")


class MetricTracker:
    """
    Track metrics across multiple runs for statistical analysis.
    """
    
    def __init__(self, metrics=None):
        """
        Parameters
        ----------
        metrics : list, optional
            Metrics to track. Default is all.
        """
        if metrics is None:
            metrics = ['ACC', 'NMI', 'ARI', 'Purity', 'F1']
        self.metrics = metrics
        self.history = {m: [] for m in metrics}
    
    def add(self, results):
        """Add results from one run."""
        for metric in self.metrics:
            if metric in results:
                self.history[metric].append(results[metric])
    
    def get_stats(self):
        """
        Get mean and std for each metric.
        
        Returns
        -------
        stats : dict
            Dictionary with 'mean' and 'std' for each metric.
        """
        stats = {}
        for metric in self.metrics:
            values = np.array(self.history[metric])
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'n_runs': len(values)
            }
        return stats
    
    def print_summary(self, dataset_name='Dataset'):
        """Print summary statistics."""
        stats = self.get_stats()
        n_runs = stats[self.metrics[0]]['n_runs'] if self.metrics else 0
        
        print(f"\n{'='*70}")
        print(f"Summary Statistics on {dataset_name} ({n_runs} runs)")
        print(f"{'='*70}")
        print(f"{'Metric':12s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
        print(f"{'-'*70}")
        
        for metric in self.metrics:
            s = stats[metric]
            print(f"{metric:12s} {s['mean']:10.4f} {s['std']:10.4f} "
                  f"{s['min']:10.4f} {s['max']:10.4f}")
        
        print(f"{'='*70}\n")
    
    def to_latex_row(self, method_name, bold_best=None):
        """
        Generate LaTeX table row.
        
        Parameters
        ----------
        method_name : str
            Name of the method.
        bold_best : dict, optional
            Dictionary mapping metric names to best values for bolding.
            
        Returns
        -------
        latex : str
            LaTeX formatted row.
        """
        stats = self.get_stats()
        cells = [method_name]
        
        for metric in self.metrics:
            mean = stats[metric]['mean']
            std = stats[metric]['std']
            
            if bold_best and metric in bold_best:
                if abs(mean - bold_best[metric]) < 1e-4:
                    cells.append(f"\\textbf{{{mean:.2f}}}$\\pm${std:.2f}")
                else:
                    cells.append(f"{mean:.2f}$\\pm${std:.2f}")
            else:
                cells.append(f"{mean:.2f}$\\pm${std:.2f}")
        
        return ' & '.join(cells) + ' \\\\'
    
    def reset(self):
        """Reset all tracked metrics."""
        self.history = {m: [] for m in self.metrics}
