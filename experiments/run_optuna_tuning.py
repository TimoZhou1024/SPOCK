"""
Optuna Hyperparameter Tuning for SPOCK

Automatically finds optimal hyperparameters for SPOCK on a given dataset.

Usage:
    python run_optuna_tuning.py --dataset Handwritten --n_trials 100
    python run_optuna_tuning.py --dataset NUSwide --n_trials 50 --metric NMI
    python run_optuna_tuning.py --dataset Caltech101-7 --n_trials 100 --timeout 3600
"""

import os
import sys
import argparse
import time
from datetime import datetime
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    print("Optuna not installed. Please run: pip install optuna")
    print("Or with uv: uv add optuna")
    sys.exit(1)

from spock import SPOCK
from spock.datasets import load_dataset
from spock.evaluation import evaluate_clustering


class SPOCKObjective:
    """
    Optuna objective function for SPOCK hyperparameter tuning.
    """
    
    def __init__(self, dataset_name, metric='ACC', n_eval_runs=3, random_seed=42):
        """
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to tune on.
        metric : str
            Metric to optimize ('ACC', 'NMI', 'ARI', 'Purity', 'F1').
        n_eval_runs : int
            Number of evaluation runs per trial for stability.
        random_seed : int
            Base random seed.
        """
        self.dataset_name = dataset_name
        self.metric = metric
        self.n_eval_runs = n_eval_runs
        self.random_seed = random_seed
        
        # Load and cache dataset
        print(f"Loading dataset: {dataset_name}")
        self.dataset = load_dataset(dataset_name)
        self.dataset.normalize('standard')
        
        self.X_views = self.dataset.views
        self.y_true = self.dataset.labels
        self.n_clusters = self.dataset.n_clusters
        self.n_samples = self.dataset.n_samples
        self.min_dim = min(v.shape[1] for v in self.X_views)
        
        print(f"Dataset info: {self.n_samples} samples, {len(self.X_views)} views, "
              f"{self.n_clusters} clusters, min_dim={self.min_dim}")
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        
        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object for suggesting hyperparameters.
            
        Returns
        -------
        score : float
            The metric score to maximize.
        """
        # Suggest hyperparameters
        params = self._suggest_params(trial)
        
        # Run multiple evaluations for stability
        scores = []
        for run in range(self.n_eval_runs):
            seed = self.random_seed + run
            try:
                model = SPOCK(
                    n_clusters=self.n_clusters,
                    random_state=seed,
                    verbose=False,
                    **params
                )
                labels = model.fit_predict(self.X_views)
                results = evaluate_clustering(self.y_true, labels)
                scores.append(results[self.metric])
            except Exception as e:
                # Return a bad score if the trial fails
                print(f"Trial {trial.number} failed: {e}")
                return 0.0
        
        # Return mean score
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Log intermediate values
        trial.set_user_attr('std', std_score)
        trial.set_user_attr('scores', scores)
        
        return mean_score
    
    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """
        Suggest hyperparameters for a trial.
        
        Defines the search space for SPOCK hyperparameters.
        """
        params = {}
        
        # Phase 1: Feature Selection
        params['alpha'] = trial.suggest_float('alpha', 0.1, 5.0, log=True)
        params['beta'] = trial.suggest_float('beta', 0.01, 0.5, log=True)
        params['lambda_l21'] = trial.suggest_float('lambda_l21', 0.001, 0.1, log=True)
        
        # Projection dimension (adaptive based on data)
        max_proj = min(self.min_dim - 1, 200)
        min_proj = max(2, min(10, max_proj // 2))
        params['proj_dim'] = trial.suggest_int('proj_dim', min_proj, max_proj)
        
        # Graph construction
        params['k_neighbors'] = trial.suggest_int('k_neighbors', 5, 30)
        
        # Landmarks for Nyström approximation
        max_landmarks = min(1000, self.n_samples // 2)
        min_landmarks = max(100, self.n_samples // 20)
        params['n_landmarks'] = trial.suggest_int('n_landmarks', min_landmarks, max_landmarks)
        
        # Phase 2: View weighting
        params['mu'] = trial.suggest_float('mu', 0.1, 0.95)
        
        # Phase 3: OT-enhanced clustering
        params['gamma'] = trial.suggest_float('gamma', 0.01, 0.3, log=True)
        params['tau'] = trial.suggest_float('tau', 0.1, 0.9)
        
        # Sinkhorn parameters
        params['sinkhorn_reg'] = trial.suggest_float('sinkhorn_reg', 0.01, 1.0, log=True)
        
        # Clustering mode
        params['use_spectral'] = trial.suggest_categorical('use_spectral', [True, False])
        
        # RFF dimension (less critical, keep fixed or narrow range)
        params['rff_dim'] = trial.suggest_categorical('rff_dim', [128, 256, 512])
        
        return params


def run_optuna_study(
    dataset_name: str,
    n_trials: int = 100,
    metric: str = 'ACC',
    n_eval_runs: int = 3,
    timeout: int = None,
    random_seed: int = 42,
    save_dir: str = './results/optuna',
    study_name: str = None,
    storage: str = None,
):
    """
    Run Optuna hyperparameter optimization for SPOCK.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    n_trials : int
        Number of optimization trials.
    metric : str
        Metric to optimize.
    n_eval_runs : int
        Number of runs per trial for stability.
    timeout : int, optional
        Maximum time in seconds for the study.
    random_seed : int
        Random seed for reproducibility.
    save_dir : str
        Directory to save results.
    study_name : str, optional
        Name for the study. Auto-generated if None.
    storage : str, optional
        Optuna storage URL (e.g., 'sqlite:///optuna.db').
        
    Returns
    -------
    study : optuna.Study
        The completed study object.
    best_params : dict
        Best hyperparameters found.
    """
    # Create objective
    objective = SPOCKObjective(
        dataset_name=dataset_name,
        metric=metric,
        n_eval_runs=n_eval_runs,
        random_seed=random_seed
    )
    
    # Study name
    if study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"SPOCK_{dataset_name}_{metric}_{timestamp}"
    
    # Create sampler and pruner
    sampler = TPESampler(seed=random_seed)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    
    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction='maximize',
        load_if_exists=True
    )
    
    print(f"\n{'='*70}")
    print(f"Starting Optuna Hyperparameter Tuning")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"Metric: {metric}")
    print(f"Trials: {n_trials}")
    print(f"Eval runs per trial: {n_eval_runs}")
    print(f"Timeout: {timeout}s" if timeout else "Timeout: None")
    print(f"Study name: {study_name}")
    print(f"{'='*70}\n")
    
    # Run optimization
    start_time = time.time()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        gc_after_trial=True
    )
    elapsed = time.time() - start_time
    
    # Get best results
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    
    print(f"\n{'='*70}")
    print(f"Optimization Complete!")
    print(f"{'='*70}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Best {metric}: {best_value:.4f}")
    print(f"Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"{'='*70}\n")
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    
    # Save best params as JSON
    results = {
        'dataset': dataset_name,
        'metric': metric,
        'best_value': best_value,
        'best_params': best_params,
        'n_trials': len(study.trials),
        'elapsed_seconds': elapsed,
        'study_name': study_name,
        'timestamp': datetime.now().isoformat()
    }
    
    json_path = os.path.join(save_dir, f'{study_name}_best.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Best params saved to: {json_path}")
    
    # Save trials history as CSV
    trials_df = study.trials_dataframe()
    csv_path = os.path.join(save_dir, f'{study_name}_trials.csv')
    trials_df.to_csv(csv_path, index=False)
    print(f"Trials history saved to: {csv_path}")
    
    # Generate Python code snippet
    code_snippet = generate_code_snippet(best_params, dataset_name)
    snippet_path = os.path.join(save_dir, f'{study_name}_code.py')
    with open(snippet_path, 'w') as f:
        f.write(code_snippet)
    print(f"Code snippet saved to: {snippet_path}")
    
    return study, best_params


def generate_code_snippet(params: dict, dataset_name: str) -> str:
    """Generate a Python code snippet with the best parameters."""
    code = f'''"""
Best SPOCK parameters for {dataset_name}
Generated by Optuna hyperparameter tuning
"""

from spock import SPOCK

# Best hyperparameters found by Optuna
best_params = {{
'''
    for k, v in params.items():
        if isinstance(v, str):
            code += f"    '{k}': '{v}',\n"
        elif isinstance(v, bool):
            code += f"    '{k}': {v},\n"
        elif isinstance(v, float):
            code += f"    '{k}': {v:.6f},\n"
        else:
            code += f"    '{k}': {v},\n"
    
    code += f'''}}

# Create SPOCK model with tuned parameters
model = SPOCK(
    n_clusters=n_clusters,  # Set based on your dataset
    random_state=42,
    verbose=True,
    **best_params
)

# Fit and predict
labels = model.fit_predict(X_views)
'''
    return code


def main():
    parser = argparse.ArgumentParser(
        description='Optuna Hyperparameter Tuning for SPOCK',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic tuning with 100 trials
  python run_optuna_tuning.py --dataset Handwritten --n_trials 100
  
  # Tune for NMI metric with timeout
  python run_optuna_tuning.py --dataset NUSwide --metric NMI --timeout 3600
  
  # Use SQLite storage for persistence
  python run_optuna_tuning.py --dataset Caltech101-7 --storage sqlite:///optuna.db
  
  # Quick test with few trials
  python run_optuna_tuning.py --dataset Handwritten --n_trials 10 --n_eval_runs 1
'''
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., Handwritten, NUSwide, Caltech101-7)')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of optimization trials (default: 100)')
    parser.add_argument('--metric', type=str, default='ACC',
                        choices=['ACC', 'NMI', 'ARI', 'Purity', 'F1'],
                        help='Metric to optimize (default: ACC)')
    parser.add_argument('--n_eval_runs', type=int, default=3,
                        help='Number of evaluation runs per trial (default: 3)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Maximum time in seconds for the study')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--save_dir', type=str, default='./results/optuna',
                        help='Directory to save results')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Name for the study (auto-generated if not provided)')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL (e.g., sqlite:///optuna.db)')
    
    args = parser.parse_args()
    
    # Suppress Optuna's default logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study, best_params = run_optuna_study(
        dataset_name=args.dataset,
        n_trials=args.n_trials,
        metric=args.metric,
        n_eval_runs=args.n_eval_runs,
        timeout=args.timeout,
        random_seed=args.seed,
        save_dir=args.save_dir,
        study_name=args.study_name,
        storage=args.storage
    )


if __name__ == '__main__':
    main()
