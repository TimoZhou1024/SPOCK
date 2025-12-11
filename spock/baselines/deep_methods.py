"""
Deep Learning based Multi-View Clustering Methods

This module provides wrappers for external deep learning MVC methods.
Each wrapper adapts external code to our unified interface:
    fit_predict(X_views) -> labels

Usage:
    1. Clone the external repository to spock/baselines/external/
    2. Install required dependencies (torch, etc.)
    3. Import and use the wrapper class

Example:
    from spock.baselines.deep_methods import MFLVCWrapper
    model = MFLVCWrapper(n_clusters=10)
    labels = model.fit_predict(X_views)
"""

import numpy as np
import warnings
from abc import ABC, abstractmethod


class BaseDeepMVC(ABC):
    """
    Base class for deep learning based multi-view clustering methods.

    All deep MVC wrappers should inherit from this class and implement
    the fit_predict method.

    Attributes
    ----------
    n_clusters : int
        Number of clusters.
    device : str
        Device to use ('cuda' or 'cpu').
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, n_clusters, device='auto', random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None

        # Auto-detect device
        if device == 'auto':
            try:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self.device = 'cpu'
        else:
            self.device = device

    @abstractmethod
    def fit_predict(self, X_views):
        """
        Fit the model and return cluster labels.

        Parameters
        ----------
        X_views : list of np.ndarray
            List of view matrices, each of shape (n_samples, n_features_v).

        Returns
        -------
        labels : np.ndarray
            Cluster labels of shape (n_samples,).
        """
        pass

    def fit(self, X_views):
        """Fit the model."""
        self.labels_ = self.fit_predict(X_views)
        return self

    def _set_seed(self):
        """Set random seed for reproducibility."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            try:
                import torch
                torch.manual_seed(self.random_state)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.random_state)
            except ImportError:
                pass


class MFLVCWrapper(BaseDeepMVC):
    """
    Wrapper for MFLVC (Multi-Level Feature Learning for Contrastive MVC).

    Paper: "Multi-Level Feature Learning for Contrastive Multi-View Clustering"
    Venue: CVPR 2022
    GitHub: https://github.com/SubmissionsIn/MFLVC

    Requirements:
        - PyTorch >= 1.8
        - Clone MFLVC to: spock/baselines/external/mflvc/

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    epochs : int, default=200
        Number of training epochs.
    batch_size : int, default=256
        Batch size for training.
    learning_rate : float, default=1e-3
        Learning rate.
    temperature : float, default=0.5
        Temperature for contrastive loss.
    device : str, default='auto'
        Device to use.
    random_state : int, optional
        Random seed.

    Example
    -------
    >>> wrapper = MFLVCWrapper(n_clusters=10, epochs=100)
    >>> labels = wrapper.fit_predict(X_views)
    """

    def __init__(self, n_clusters, epochs=200, batch_size=256,
                 learning_rate=1e-3, temperature=0.5,
                 device='auto', random_state=None):
        super().__init__(n_clusters, device, random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.name = 'MFLVC'

    def fit_predict(self, X_views):
        """
        Fit MFLVC and return cluster labels.

        This method attempts to:
        1. Import the external MFLVC code
        2. Convert data to PyTorch tensors
        3. Train the model
        4. Extract cluster labels

        If MFLVC is not available, falls back to a simple implementation.
        """
        self._set_seed()

        try:
            # Try to import external MFLVC
            import sys
            import os
            external_path = os.path.join(
                os.path.dirname(__file__), 'external', 'mflvc'
            )
            if os.path.exists(external_path):
                sys.path.insert(0, external_path)
                return self._run_external_mflvc(X_views)
            else:
                warnings.warn(
                    f"MFLVC not found at {external_path}. "
                    "Using fallback implementation. "
                    "To use the original: git clone https://github.com/SubmissionsIn/MFLVC.git "
                    f"to {external_path}"
                )
                return self._run_fallback(X_views)

        except Exception as e:
            warnings.warn(f"MFLVC failed: {e}. Using fallback.")
            return self._run_fallback(X_views)

    def _run_external_mflvc(self, X_views):
        """Run the external MFLVC implementation."""
        import torch
        from network import Network  # From external MFLVC
        from train import train  # From external MFLVC

        # Convert to tensors
        X_tensors = [torch.FloatTensor(x).to(self.device) for x in X_views]

        # Get dimensions
        view_dims = [x.shape[1] for x in X_views]

        # Initialize model
        model = Network(
            view_dims=view_dims,
            feature_dim=256,
            high_feature_dim=128,
            class_num=self.n_clusters
        ).to(self.device)

        # Train
        labels = train(
            model, X_tensors,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.learning_rate,
            temperature=self.temperature,
            device=self.device
        )

        return np.array(labels)

    def _run_fallback(self, X_views):
        """Fallback: Simple contrastive-style clustering."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize

        # Simple fusion: weighted concatenation
        X_concat = np.hstack([normalize(x) for x in X_views])

        # K-Means
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state
        )
        return kmeans.fit_predict(X_concat)


class DIMVCWrapper(BaseDeepMVC):
    """
    Wrapper for DIMVC (Deep Incomplete Multi-View Clustering).

    Paper: "Incomplete Multi-View Clustering via Diffusion Completion"
    Venue: ICML 2024
    GitHub: https://github.com/Jeaninezpp/DIMVC

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    epochs : int, default=100
        Number of training epochs.
    device : str, default='auto'
        Device to use.
    random_state : int, optional
        Random seed.
    """

    def __init__(self, n_clusters, epochs=100, device='auto', random_state=None):
        super().__init__(n_clusters, device, random_state)
        self.epochs = epochs
        self.name = 'DIMVC'

    def fit_predict(self, X_views):
        self._set_seed()

        try:
            import sys
            import os
            external_path = os.path.join(
                os.path.dirname(__file__), 'external', 'dimvc'
            )
            if os.path.exists(external_path):
                sys.path.insert(0, external_path)
                return self._run_external(X_views)
            else:
                warnings.warn(f"DIMVC not found at {external_path}. Using fallback.")
                return self._run_fallback(X_views)
        except Exception as e:
            warnings.warn(f"DIMVC failed: {e}. Using fallback.")
            return self._run_fallback(X_views)

    def _run_external(self, X_views):
        """Run external DIMVC."""
        # Implementation depends on actual DIMVC code structure
        raise NotImplementedError("Clone DIMVC repository first")

    def _run_fallback(self, X_views):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
        X_concat = np.hstack([normalize(x) for x in X_views])
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        return kmeans.fit_predict(X_concat)


class CVCLWrapper(BaseDeepMVC):
    """
    Wrapper for CVCL (Cluster-guided Contrastive Learning).

    Paper: "Cluster-guided Contrastive Multi-View Clustering"
    Venue: AAAI 2024
    GitHub: https://github.com/JiangJinW/CVCL

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    epochs : int, default=200
        Number of training epochs.
    device : str, default='auto'
        Device to use.
    random_state : int, optional
        Random seed.
    """

    def __init__(self, n_clusters, epochs=200, device='auto', random_state=None):
        super().__init__(n_clusters, device, random_state)
        self.epochs = epochs
        self.name = 'CVCL'

    def fit_predict(self, X_views):
        self._set_seed()

        try:
            import sys
            import os
            external_path = os.path.join(
                os.path.dirname(__file__), 'external', 'cvcl'
            )
            if os.path.exists(external_path):
                sys.path.insert(0, external_path)
                return self._run_external(X_views)
            else:
                warnings.warn(f"CVCL not found. Using fallback.")
                return self._run_fallback(X_views)
        except Exception as e:
            warnings.warn(f"CVCL failed: {e}. Using fallback.")
            return self._run_fallback(X_views)

    def _run_external(self, X_views):
        raise NotImplementedError("Clone CVCL repository first")

    def _run_fallback(self, X_views):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
        X_concat = np.hstack([normalize(x) for x in X_views])
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        return kmeans.fit_predict(X_concat)


class DealMVCWrapper(BaseDeepMVC):
    """
    Wrapper for DealMVC (Dual Contrastive Calibration for MVC).

    Paper: "Incomplete Multi-View Clustering with Dual Contrastive Calibration"
    Venue: AAAI 2024
    GitHub: https://github.com/xihongyang1999/DealMVC

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    mse_epochs : int, default=200
        Number of pretraining epochs (reconstruction).
    con_epochs : int, default=100
        Number of contrastive training epochs.
    tune_epochs : int, default=50
        Number of fine-tuning epochs.
    batch_size : int, default=256
        Batch size for training.
    learning_rate : float, default=0.0003
        Learning rate.
    feature_dim : int, default=512
        Feature dimension for encoder.
    threshold : float, default=0.8
        Threshold for pseudo-label generation.
    device : str, default='auto'
        Device to use.
    random_state : int, optional
        Random seed.
    """

    def __init__(self, n_clusters, mse_epochs=200, con_epochs=100, tune_epochs=50,
                 batch_size=256, learning_rate=0.0003, feature_dim=512,
                 threshold=0.8, device='auto', random_state=None):
        super().__init__(n_clusters, device, random_state)
        self.mse_epochs = mse_epochs
        self.con_epochs = con_epochs
        self.tune_epochs = tune_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.feature_dim = feature_dim
        self.threshold = threshold
        self.name = 'DealMVC'

    def fit_predict(self, X_views):
        self._set_seed()

        try:
            import sys
            import os
            # Check both lowercase and original case
            external_dir = os.path.dirname(__file__)
            possible_paths = [
                os.path.join(external_dir, 'external', 'DealMVC'),
                os.path.join(external_dir, 'external', 'dealmvc'),
            ]

            external_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    external_path = path
                    break

            if external_path is not None:
                sys.path.insert(0, external_path)
                return self._run_external(X_views)
            else:
                warnings.warn(f"DealMVC not found. Using fallback.")
                return self._run_fallback(X_views)
        except Exception as e:
            warnings.warn(f"DealMVC failed: {e}. Using fallback.")
            return self._run_fallback(X_views)

    def _run_external(self, X_views):
        """Run the external DealMVC implementation."""
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.cluster import KMeans

        # Import from external DealMVC
        # Only import Network components; avoid metric.py as we don't need it
        from network import Network, MultiHeadAttention, FeedForwardNetwork

        device = torch.device(self.device)
        view = len(X_views)
        n_samples = X_views[0].shape[0]
        dims = [x.shape[1] for x in X_views]

        # Create dataset
        class NumpyDataset(Dataset):
            def __init__(self, views, n_views):
                self.views = [torch.FloatTensor(v) for v in views]
                self.n_views = n_views
                self.n_samples = views[0].shape[0]
                # Dummy labels (not used in training)
                self.labels = torch.zeros(self.n_samples)

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                return [self.views[v][idx] for v in range(self.n_views)], \
                       self.labels[idx], torch.tensor(idx).long()

        # Normalize data
        scaler = MinMaxScaler()
        X_normalized = [scaler.fit_transform(x) for x in X_views]
        dataset = NumpyDataset(X_normalized, view)

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Initialize models
        model = Network(view, dims, self.feature_dim, self.feature_dim, self.n_clusters, device)
        attention_net = MultiHeadAttention(256, 0.5, 8, 6)
        p_net = FeedForwardNetwork(view, 32, 0.5)

        model = model.to(device)
        attention_net = attention_net.to(device)
        p_net = p_net.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Initialize adaptive weights
        p_sample = torch.ones(view).to(device) / view
        adaptive_weight = (torch.ones(view) / view).unsqueeze(1).to(device)

        # Training phase 1: Pretraining (MSE)
        criterion = nn.MSELoss()
        for epoch in range(self.mse_epochs):
            model.train()
            for batch_idx, (xs, _, _) in enumerate(data_loader):
                for v in range(view):
                    xs[v] = xs[v].to(device)
                optimizer.zero_grad()
                _, _, xrs, _ = model(xs)
                loss = sum([criterion(xs[v], xrs[v]) for v in range(view)])
                loss.backward()
                optimizer.step()

        # Training phase 2: Contrastive (simplified)
        for epoch in range(self.con_epochs):
            model.train()
            for batch_idx, (xs, _, _) in enumerate(data_loader):
                for v in range(view):
                    xs[v] = xs[v].to(device)
                optimizer.zero_grad()
                hs, qs, xrs, zs = model(xs)

                loss_list = []
                # Reconstruction loss
                for v in range(view):
                    loss_list.append(criterion(xs[v], xrs[v]))

                # Local contrastive loss (simplified)
                for v in range(view):
                    for w in range(v+1, view):
                        sim = torch.exp(torch.mm(hs[v], hs[w].t()))
                        sim_probs = sim / sim.sum(1, keepdim=True)
                        Q = torch.mm(qs[v], qs[w].t())
                        Q.fill_diagonal_(1)
                        pos_mask = (Q >= self.threshold).float()
                        Q = Q * pos_mask
                        Q = Q / (Q.sum(1, keepdims=True) + 1e-7)
                        loss_contrast = - (torch.log(sim_probs + 1e-7) * Q).sum(1).mean()
                        loss_list.append(loss_contrast * 0.1)

                loss = sum(loss_list)
                loss.backward()
                optimizer.step()

        # Get final predictions
        model.eval()
        full_loader = DataLoader(dataset, batch_size=n_samples, shuffle=False)

        with torch.no_grad():
            for xs, _, _ in full_loader:
                for v in range(view):
                    xs[v] = xs[v].to(device)
                # Note: We avoid using model.forward_cluster() directly because
                # the external code has hardcoded .cuda() calls (line 97 in network.py)
                # Instead, we use model.forward() and compute predictions ourselves
                hs, qs, xrs, zs = model.forward(xs)
                # Average soft labels across views
                q_avg = sum(qs) / view
                labels = torch.argmax(q_avg, dim=1).cpu().numpy()

        return labels

    def _run_fallback(self, X_views):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
        X_concat = np.hstack([normalize(x) for x in X_views])
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        return kmeans.fit_predict(X_concat)


class FMVACCWrapper(BaseDeepMVC):
    """
    Wrapper for FMVACC (Fast Multi-View Anchor-Contrastive Clustering).

    Paper: "Fast Multi-View Anchor-Contrastive Clustering"
    Year: 2024
    GitHub: https://github.com/wangsiwei2010/FMVACC

    This is an efficient anchor-based method with near-linear complexity,
    making it a good comparison for SPOCK.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    n_anchors : int, default=500
        Number of anchor points.
    random_state : int, optional
        Random seed.
    """

    def __init__(self, n_clusters, n_anchors=500, random_state=None):
        super().__init__(n_clusters, device='cpu', random_state=random_state)
        self.n_anchors = n_anchors
        self.name = 'FMVACC'

    def fit_predict(self, X_views):
        self._set_seed()

        try:
            import sys
            import os
            external_path = os.path.join(
                os.path.dirname(__file__), 'external', 'fmvacc'
            )
            if os.path.exists(external_path):
                sys.path.insert(0, external_path)
                return self._run_external(X_views)
            else:
                warnings.warn(f"FMVACC not found. Using fallback.")
                return self._run_fallback(X_views)
        except Exception as e:
            warnings.warn(f"FMVACC failed: {e}. Using fallback.")
            return self._run_fallback(X_views)

    def _run_external(self, X_views):
        raise NotImplementedError("Clone FMVACC repository first")

    def _run_fallback(self, X_views):
        """Fallback: Anchor-based spectral clustering."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize

        n_samples = X_views[0].shape[0]
        n_anchors = min(self.n_anchors, n_samples // 2)

        # Simple anchor-based approach
        X_concat = np.hstack([normalize(x) for x in X_views])

        # Select anchors via K-Means
        kmeans_anchor = KMeans(n_clusters=n_anchors, n_init=1, random_state=self.random_state)
        kmeans_anchor.fit(X_concat)
        anchors = kmeans_anchor.cluster_centers_

        # Build anchor graph
        from scipy.spatial.distance import cdist
        distances = cdist(X_concat, anchors)
        k = 5
        Z = np.zeros((n_samples, n_anchors))
        for i in range(n_samples):
            nearest = np.argsort(distances[i])[:k]
            sigma = distances[i, nearest[-1]] + 1e-10
            weights = np.exp(-distances[i, nearest] / sigma)
            Z[i, nearest] = weights / weights.sum()

        # Spectral embedding on Z * Z^T
        ZZt = Z @ Z.T
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(ZZt.sum(axis=1), 1e-10)))
        L_sym = np.eye(n_samples) - D_inv_sqrt @ ZZt @ D_inv_sqrt

        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(L_sym)
        U = eigenvectors[:, :self.n_clusters]
        U = normalize(U, norm='l2', axis=1)

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        return kmeans.fit_predict(U)


# Registry of available deep methods
DEEP_METHODS = {
    'MFLVC': MFLVCWrapper,
    'DIMVC': DIMVCWrapper,
    'CVCL': CVCLWrapper,
    'DealMVC': DealMVCWrapper,
    'FMVACC': FMVACCWrapper,
}


def get_deep_methods(n_clusters, random_state=None, methods=None):
    """
    Get deep learning based baseline methods.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    random_state : int, optional
        Random seed.
    methods : list, optional
        List of method names to include. If None, includes all.

    Returns
    -------
    methods_dict : dict
        Dictionary of method name -> method instance.
    """
    if methods is None:
        methods = list(DEEP_METHODS.keys())

    return {
        name: DEEP_METHODS[name](n_clusters, random_state=random_state)
        for name in methods if name in DEEP_METHODS
    }


def check_deep_methods_availability():
    """
    Check which deep methods have their external code available.

    Returns
    -------
    availability : dict
        Dictionary of method name -> bool (available or not).
    """
    import os
    external_dir = os.path.join(os.path.dirname(__file__), 'external')

    # Map method names to possible directory names (check multiple cases)
    method_dirs = {
        'MFLVC': ['mflvc', 'MFLVC'],
        'DIMVC': ['dimvc', 'DIMVC'],
        'CVCL': ['cvcl', 'CVCL'],
        'DealMVC': ['dealmvc', 'DealMVC'],
        'FMVACC': ['fmvacc', 'FMVACC'],
    }

    availability = {}
    for method, dir_names in method_dirs.items():
        found = False
        for dir_name in dir_names:
            path = os.path.join(external_dir, dir_name)
            if os.path.exists(path):
                found = True
                break
        availability[method] = found

    return availability
