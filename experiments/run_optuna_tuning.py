"""
Optuna Hyperparameter Tuning for SPOCK

Automatically finds optimal hyperparameters for SPOCK on a given dataset.
Supports robustness-aware tuning for incomplete and unaligned data scenarios.

Usage:
    # Standard tuning
    python run_optuna_tuning.py --dataset Handwritten --n_trials 100

    # Robustness-aware tuning
    python run_optuna_tuning.py --dataset Handwritten --n_trials 100 --robustness both

    # With specific metric and timeout
    python run_optuna_tuning.py --dataset NUSwide --metric NMI --timeout 3600
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    from optuna.trial import Trial
except ImportError:
    print("Optuna not installed. Please run: pip install optuna")
    print("Or with uv: uv add optuna")
    sys.exit(1)

from spock import SPOCK
from spock.datasets import load_dataset
from spock.evaluation import evaluate_clustering


# ============================================================
# Constants and Configuration
# ============================================================

# Robustness tuning settings
ROBUSTNESS_MISSING_RATES = [0.0, 0.1, 0.3, 0.5]
ROBUSTNESS_UNALIGNED_RATES = [0.0, 0.2, 0.4]

# Path configurations
CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'config'
)
TUNED_PARAMS_PATH = os.path.join(CONFIG_DIR, 'tuned_params.json')
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results', 'optuna'
)


# ============================================================
# Helper Functions for Missing/Unaligned Data
# ============================================================

def create_missing_views(views: List[np.ndarray], missing_rate: float,
                         random_state: int = None) -> tuple:
    """Create incomplete multi-view data by setting missing views to zeros."""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = views[0].shape[0]
    n_views = len(views)

    mask = np.ones((n_samples, n_views), dtype=bool)
    n_missing_samples = int(missing_rate * n_samples)
    missing_sample_idx = np.random.choice(n_samples, n_missing_samples, replace=False)

    for idx in missing_sample_idx:
        n_remove = np.random.randint(1, n_views)
        remove_views = np.random.choice(n_views, n_remove, replace=False)
        mask[idx, remove_views] = False

    incomplete_views = []
    for v_idx, view in enumerate(views):
        new_view = view.copy()
        missing_mask = ~mask[:, v_idx]
        new_view[missing_mask] = 0
        incomplete_views.append(new_view)

    return incomplete_views, mask


def create_unaligned_views(views: List[np.ndarray], shuffle_rate: float,
                           random_state: int = None) -> tuple:
    """Create unaligned multi-view data by shuffling samples in non-first views."""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = views[0].shape[0]
    unaligned_views = [views[0].copy()]

    for v_idx in range(1, len(views)):
        new_view = views[v_idx].copy()
        if shuffle_rate > 0:
            n_shuffle = int(shuffle_rate * n_samples)
            shuffle_idx = np.random.choice(n_samples, n_shuffle, replace=False)
            shuffled_positions = np.random.permutation(shuffle_idx)
            new_view[shuffle_idx] = views[v_idx][shuffled_positions]
        unaligned_views.append(new_view)

    return unaligned_views, {'shuffle_rate': shuffle_rate}


def impute_missing_views(views: List[np.ndarray], mask: np.ndarray) -> List[np.ndarray]:
    """Simple mean imputation for missing views."""
    imputed_views = []
    for v_idx, view in enumerate(views):
        new_view = view.copy()
        missing_mask = ~mask[:, v_idx]
        available_mask = mask[:, v_idx]
        if available_mask.sum() > 0:
            view_mean = view[available_mask].mean(axis=0)
            new_view[missing_mask] = view_mean
        imputed_views.append(new_view)
    return imputed_views


# ============================================================
# SPOCKHyperparameterTuner Class
# ============================================================

class SPOCKHyperparameterTuner:
    """
    Optuna hyperparameter tuner for SPOCK with robustness mode support.

    Supports standard tuning and robustness-aware tuning for incomplete
    and unaligned data scenarios.
    """

    def __init__(
        self,
        dataset_name: str,
        metric: str = 'ACC',
        n_eval_runs: int = 3,
        robustness_mode: str = 'none',
        random_seed: int = 42,
        device: str = 'auto'
    ):
        """
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to tune on.
        metric : str
            Metric to optimize ('ACC', 'NMI', 'ARI', 'Purity', 'F1').
        n_eval_runs : int
            Number of evaluation runs per trial for stability.
        robustness_mode : str
            Robustness tuning mode: 'none', 'incomplete', 'unaligned', 'both'.
        random_seed : int
            Base random seed.
        device : str
            Device for computation ('auto', 'cuda', 'cpu').
        """
        self.dataset_name = dataset_name
        self.metric = metric
        self.n_eval_runs = n_eval_runs
        self.robustness_mode = robustness_mode
        self.random_seed = random_seed
        self.device = device

        # Auto-detect device
        self._detect_device()

        # Load and cache dataset
        self._load_dataset()

    def _detect_device(self):
        """Auto-detect best available device."""
        if self.device == 'auto':
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = 'mps'
                else:
                    self.device = 'cpu'
            except ImportError:
                self.device = 'cpu'

    def _load_dataset(self):
        """Load and cache dataset."""
        print(f"Loading dataset: {self.dataset_name}")
        self.dataset = load_dataset(self.dataset_name)
        self.dataset.normalize('standard')

        self.X_views = self.dataset.views
        self.y_true = self.dataset.labels
        self.n_clusters = self.dataset.n_clusters
        self.n_samples = self.dataset.n_samples
        self.min_dim = min(v.shape[1] for v in self.X_views)

        print(f"Dataset info: {self.n_samples} samples, {len(self.X_views)} views, "
              f"{self.n_clusters} clusters, min_dim={self.min_dim}")
    
    def __call__(self, trial: Trial) -> float:
        """Optuna objective function."""
        if self.robustness_mode == 'none':
            return self._standard_objective(trial)
        else:
            return self._robustness_objective(trial)

    def _standard_objective(self, trial: Trial) -> float:
        """Standard objective function (no robustness)."""
        params = self._suggest_params(trial)

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
                print(f"Trial {trial.number} failed: {e}")
                return 0.0

        mean_score = np.mean(scores)
        trial.set_user_attr('std', np.std(scores))
        trial.set_user_attr('scores', scores)
        return mean_score

    def _robustness_objective(self, trial: Trial) -> float:
        """Robustness-aware objective function."""
        params = self._suggest_params(trial)
        all_scores = []

        # Determine conditions based on robustness mode
        if self.robustness_mode == 'incomplete':
            conditions = [(mr, 0.0) for mr in ROBUSTNESS_MISSING_RATES]
        elif self.robustness_mode == 'unaligned':
            conditions = [(0.0, ur) for ur in ROBUSTNESS_UNALIGNED_RATES]
        else:  # 'both'
            conditions = [
                (0.0, 0.0), (0.3, 0.0), (0.5, 0.0),
                (0.0, 0.2), (0.0, 0.4), (0.3, 0.2),
            ]

        for missing_rate, unaligned_rate in conditions:
            try:
                score = self._evaluate_condition(
                    params, missing_rate, unaligned_rate
                )
                all_scores.append(score)

                # Report for pruning
                trial.report(np.mean(all_scores), len(all_scores) - 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            except optuna.TrialPruned:
                raise
            except Exception as e:
                print(f"  Condition failed: {e}")
                all_scores.append(0.0)

        if not all_scores:
            return 0.0

        # Weight harder conditions more
        weights = np.linspace(1.0, 1.5, len(all_scores))
        weighted_score = np.average(all_scores, weights=weights)
        trial.set_user_attr('condition_scores', all_scores)
        return weighted_score

    def _evaluate_condition(self, params: dict, missing_rate: float,
                            unaligned_rate: float) -> float:
        """Evaluate model under specific condition."""
        views = self.X_views

        # Apply missing views
        if missing_rate > 0:
            views, mask = create_missing_views(views, missing_rate, self.random_seed)
            views = impute_missing_views(views, mask)

        # Apply unaligned views
        if unaligned_rate > 0:
            views, _ = create_unaligned_views(views, unaligned_rate, self.random_seed)

        model = SPOCK(
            n_clusters=self.n_clusters,
            random_state=self.random_seed,
            verbose=False,
            **params
        )
        labels = model.fit_predict(views)
        results = evaluate_clustering(self.y_true, labels)
        return results[self.metric]
    
    def _suggest_params(self, trial: Trial) -> dict:
        """Suggest hyperparameters for a trial."""
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

    def run(
        self,
        n_trials: int = 100,
        timeout: int = None,
        storage: str = None,
        study_name: str = None,
        save_dir: str = None
    ) -> tuple:
        """
        Run the hyperparameter optimization study.

        Returns
        -------
        study : optuna.Study
        best_params : dict
        """
        if save_dir is None:
            save_dir = RESULTS_DIR

        # Generate study name
        if study_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.robustness_mode != 'none':
                study_name = f"SPOCK_{self.dataset_name}_{self.robustness_mode}_{timestamp}"
            else:
                study_name = f"SPOCK_{self.dataset_name}_{self.metric}_{timestamp}"

        # Create sampler and pruner
        sampler = TPESampler(seed=self.random_seed)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

        # Create study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            direction='maximize',
            load_if_exists=True
        )

        # Print info
        print(f"\n{'='*60}")
        print(f"Starting Optuna Hyperparameter Tuning")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Metric: {self.metric}")
        print(f"Robustness mode: {self.robustness_mode}")
        print(f"Trials: {n_trials}")
        print(f"Device: {self.device}")
        if self.robustness_mode != 'none':
            if self.robustness_mode == 'incomplete':
                print(f"  Missing rates: {ROBUSTNESS_MISSING_RATES}")
            elif self.robustness_mode == 'unaligned':
                print(f"  Unaligned rates: {ROBUSTNESS_UNALIGNED_RATES}")
            else:
                print(f"  Testing combined conditions")
        print(f"{'='*60}\n")

        # Run optimization
        start_time = time.time()
        study.optimize(
            self,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            gc_after_trial=True
        )
        elapsed = time.time() - start_time

        # Get best results
        best_params = study.best_params
        best_value = study.best_value

        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}")
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Best {self.metric}: {best_value:.4f}")
        print(f"Best parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}\n")

        # Save results
        self._save_results(study, best_params, best_value, save_dir, elapsed)

        return study, best_params

    def _save_results(self, study, best_params: dict, best_value: float,
                      save_dir: str, elapsed: float):
        """Save tuning results."""
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(CONFIG_DIR, exist_ok=True)

        # Update tuned_params.json
        if os.path.exists(TUNED_PARAMS_PATH):
            with open(TUNED_PARAMS_PATH, 'r') as f:
                all_params = json.load(f)
        else:
            all_params = {}

        # Determine key
        key = self.dataset_name.lower()
        if self.robustness_mode != 'none':
            key = f"{key}_robustness_{self.robustness_mode}"

        all_params[key] = {
            'params': best_params,
            'best_value': best_value,
            'metric': self.metric,
            'robustness_mode': self.robustness_mode,
            'n_trials': len(study.trials),
            'timestamp': datetime.now().isoformat()
        }

        with open(TUNED_PARAMS_PATH, 'w') as f:
            json.dump(all_params, f, indent=2)
        print(f"Tuned params saved to: {TUNED_PARAMS_PATH}")

        # Save study JSON
        study_name = study.study_name
        json_path = os.path.join(save_dir, f'{study_name}_best.json')
        results = {
            'dataset': self.dataset_name,
            'metric': self.metric,
            'robustness_mode': self.robustness_mode,
            'best_value': best_value,
            'best_params': best_params,
            'n_trials': len(study.trials),
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat()
        }
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Study results saved to: {json_path}")

        # Save trials CSV
        trials_df = study.trials_dataframe()
        csv_path = os.path.join(save_dir, f'{study_name}_trials.csv')
        trials_df.to_csv(csv_path, index=False)
        print(f"Trials history saved to: {csv_path}")


# ============================================================
# Helper Functions
# ============================================================

def load_tuned_params(
    dataset_name: str,
    tuned_key: str = None,
    config_dir: str = None
) -> Optional[Dict[str, Any]]:
    """
    Load Optuna-tuned parameters for a dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    tuned_key : str, optional
        Specific tuning key (e.g., 'robustness_incomplete').
    config_dir : str, optional
        Config directory path.

    Returns
    -------
    params : dict or None
    """
    params_path = TUNED_PARAMS_PATH
    if config_dir:
        params_path = os.path.join(config_dir, 'tuned_params.json')

    if not os.path.exists(params_path):
        return None

    try:
        with open(params_path, 'r') as f:
            all_params = json.load(f)

        # Determine key
        if tuned_key:
            key = f"{dataset_name.lower()}_{tuned_key}"
        else:
            key = dataset_name.lower()

        if key not in all_params:
            print(f"No tuned params for key: {key}")
            print(f"Available keys: {list(all_params.keys())}")
            return None

        return all_params[key]['params']
    except Exception as e:
        print(f"Warning: Could not load tuned params: {e}")
        return None


def apply_tuned_params(
    n_clusters: int,
    dataset_name: str,
    tuned_key: str = None,
    random_state: int = None,
    **override_params
) -> SPOCK:
    """
    Create a SPOCK model with tuned parameters.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    dataset_name : str
        Dataset name for loading tuned params.
    tuned_key : str, optional
        Specific tuning key.
    random_state : int, optional
        Random seed.
    **override_params
        Parameters to override tuned values.

    Returns
    -------
    model : SPOCK
    """
    params = load_tuned_params(dataset_name, tuned_key) or {}
    params.update(override_params)

    return SPOCK(
        n_clusters=n_clusters,
        random_state=random_state,
        **params
    )


def run_optuna_study(
    dataset_name: str,
    n_trials: int = 100,
    metric: str = 'ACC',
    n_eval_runs: int = 3,
    timeout: int = None,
    random_seed: int = 42,
    save_dir: str = None,
    study_name: str = None,
    storage: str = None,
    robustness_mode: str = 'none',
    device: str = 'auto',
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
        Name for the study.
    storage : str, optional
        Optuna storage URL.
    robustness_mode : str
        Robustness tuning mode: 'none', 'incomplete', 'unaligned', 'both'.
    device : str
        Device for computation.

    Returns
    -------
    study : optuna.Study
    best_params : dict
    """
    tuner = SPOCKHyperparameterTuner(
        dataset_name=dataset_name,
        metric=metric,
        n_eval_runs=n_eval_runs,
        robustness_mode=robustness_mode,
        random_seed=random_seed,
        device=device
    )

    return tuner.run(
        n_trials=n_trials,
        timeout=timeout,
        storage=storage,
        study_name=study_name,
        save_dir=save_dir
    )


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
  # Standard tuning
  python run_optuna_tuning.py --dataset Handwritten --n_trials 100

  # Robustness-aware tuning for incomplete data
  python run_optuna_tuning.py --dataset Handwritten --robustness incomplete

  # Robustness-aware tuning for both conditions
  python run_optuna_tuning.py --dataset Handwritten --robustness both

  # Quick test
  python run_optuna_tuning.py --dataset Handwritten --n_trials 10 --n_eval_runs 1

Usage hints for robustness-tuned parameters:
  python run_robustness.py --dataset Handwritten --use_tuned --tuned_key robustness_both
'''
    )

    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of optimization trials')
    parser.add_argument('--metric', type=str, default='ACC',
                        choices=['ACC', 'NMI', 'ARI', 'Purity', 'F1'],
                        help='Metric to optimize')
    parser.add_argument('--n_eval_runs', type=int, default=3,
                        help='Number of evaluation runs per trial')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Maximum time in seconds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Name for the study')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL')
    parser.add_argument('--robustness', type=str, default='none',
                        choices=['none', 'incomplete', 'unaligned', 'both'],
                        help='Robustness tuning mode')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Device for computation')

    args = parser.parse_args()

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
        storage=args.storage,
        robustness_mode=args.robustness,
        device=args.device
    )

    # Print usage hints
    if args.robustness != 'none':
        key = f"robustness_{args.robustness}"
        print(f"\nTo use robustness-tuned parameters:")
        print(f"  python run_robustness.py --dataset {args.dataset} --use_tuned --tuned_key {key}")


if __name__ == '__main__':
    main()
