"""
Multi-View Dataset Loaders

Supports various benchmark datasets for multi-view clustering evaluation.
"""

import os
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import urllib.request
import zipfile


class MultiViewDataset:
    """
    Base class for multi-view datasets.
    
    Attributes
    ----------
    views : list of ndarray
        List of feature matrices for each view.
    labels : ndarray
        Ground truth labels.
    n_views : int
        Number of views.
    n_samples : int
        Number of samples.
    n_clusters : int
        Number of clusters.
    name : str
        Dataset name.
    """
    
    def __init__(self, name='unknown'):
        self.views = []
        self.labels = None
        self.n_views = 0
        self.n_samples = 0
        self.n_clusters = 0
        self.name = name
    
    def normalize(self, method='standard'):
        """
        Normalize each view.
        
        Parameters
        ----------
        method : str
            'standard' (z-score), 'minmax', or 'l2'
        """
        for i in range(self.n_views):
            if method == 'standard':
                scaler = StandardScaler()
                self.views[i] = scaler.fit_transform(self.views[i])
            elif method == 'minmax':
                scaler = MinMaxScaler()
                self.views[i] = scaler.fit_transform(self.views[i])
            elif method == 'l2':
                norms = np.linalg.norm(self.views[i], axis=1, keepdims=True)
                self.views[i] = self.views[i] / (norms + 1e-10)
        
        return self
    
    def add_missing(self, missing_rate=0.3, random_state=None):
        """
        Simulate missing views.
        
        Parameters
        ----------
        missing_rate : float
            Fraction of samples to have missing views.
        random_state : int, optional
            Random seed.
            
        Returns
        -------
        mask : ndarray of shape (n_samples, n_views)
            Boolean mask where True indicates available view.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        mask = np.ones((self.n_samples, self.n_views), dtype=bool)
        
        for i in range(self.n_samples):
            if np.random.rand() < missing_rate:
                # Randomly remove some views (keep at least one)
                n_remove = np.random.randint(1, self.n_views)
                remove_idx = np.random.choice(self.n_views, n_remove, replace=False)
                mask[i, remove_idx] = False
        
        return mask
    
    def get_view_dimensions(self):
        """Get dimensions of each view."""
        return [v.shape[1] for v in self.views]
    
    def __repr__(self):
        dims = self.get_view_dimensions()
        return (f"MultiViewDataset({self.name}): "
                f"{self.n_samples} samples, {self.n_views} views, "
                f"{self.n_clusters} clusters, dims={dims}")


def load_handwritten(data_path='./data'):
    """
    Load Handwritten Digits dataset (UCI).
    
    6 views: 76-dim profile correlations, 216-dim Fourier coefficients,
    64-dim Karhunen-Love coefficients, 240-dim pixel averages,
    47-dim Zernike moments, 6-dim morphological features.
    
    2000 samples, 10 classes (digits 0-9), 200 per class.
    """
    dataset = MultiViewDataset('Handwritten')
    
    file_path = os.path.join(data_path, 'handwritten.mat')
    
    if not os.path.exists(file_path):
        # Create synthetic data for testing
        print(f"Warning: {file_path} not found. Generating synthetic data.")
        return _generate_synthetic_multiview(
            n_samples=2000, n_clusters=10, n_views=6, 
            view_dims=[76, 216, 64, 240, 47, 6], name='Handwritten'
        )
    
    data = loadmat(file_path)
    
    # Different mat file formats
    if 'X' in data:
        views = data['X'].flatten()
        dataset.views = [np.array(v) for v in views]
        dataset.labels = data['Y'].flatten()
    elif 'fea' in data:
        views = data['fea'].flatten()
        dataset.views = [np.array(v) for v in views]
        dataset.labels = data['gnd'].flatten()
    else:
        # Try common patterns
        for key in data.keys():
            if key.startswith('x') or key.startswith('X'):
                dataset.views.append(np.array(data[key]))
        dataset.labels = data.get('Y', data.get('gnd', data.get('gt'))).flatten()
    
    dataset.labels = dataset.labels.astype(int)
    if dataset.labels.min() == 1:
        dataset.labels -= 1  # Convert to 0-indexed
    
    dataset.n_views = len(dataset.views)
    dataset.n_samples = dataset.views[0].shape[0]
    dataset.n_clusters = len(np.unique(dataset.labels))
    
    return dataset


def load_caltech101(data_path='./data', n_classes=7):
    """
    Load Caltech-101 dataset (subset).
    
    6 views: Gabor, Wavelet moment, CENTRIST, HOG, GIST, LBP.
    
    Parameters
    ----------
    n_classes : int
        Number of classes to use (7 or 20 commonly used).
    """
    dataset = MultiViewDataset(f'Caltech101-{n_classes}')
    
    file_path = os.path.join(data_path, f'Caltech101_{n_classes}.mat')
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Generating synthetic data.")
        return _generate_synthetic_multiview(
            n_samples=1474, n_clusters=n_classes, n_views=6,
            view_dims=[48, 40, 254, 1984, 512, 928],
            name=f'Caltech101-{n_classes}'
        )
    
    data = loadmat(file_path)
    
    # Extract views and labels
    if 'X' in data:
        views = data['X'].flatten()
        dataset.views = [np.array(v) for v in views]
        dataset.labels = data['Y'].flatten().astype(int)
    else:
        for key in sorted(data.keys()):
            if 'fea' in key.lower() or key.startswith('X'):
                dataset.views.append(np.array(data[key]))
        dataset.labels = data.get('Y', data.get('gnd')).flatten().astype(int)
    
    if dataset.labels.min() == 1:
        dataset.labels -= 1
    
    dataset.n_views = len(dataset.views)
    dataset.n_samples = dataset.views[0].shape[0]
    dataset.n_clusters = len(np.unique(dataset.labels))
    
    return dataset


def load_scene15(data_path='./data'):
    """
    Load Scene-15 dataset.
    
    3 views: PHOG, Gist, LBP.
    4485 samples, 15 scene categories.
    """
    dataset = MultiViewDataset('Scene15')
    
    file_path = os.path.join(data_path, 'Scene15.mat')
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Generating synthetic data.")
        return _generate_synthetic_multiview(
            n_samples=4485, n_clusters=15, n_views=3,
            view_dims=[59, 20, 40], name='Scene15'
        )
    
    data = loadmat(file_path)
    
    if 'X' in data:
        views = data['X'].flatten()
        dataset.views = [np.array(v) for v in views]
        dataset.labels = data['Y'].flatten().astype(int)
    else:
        for key in sorted(data.keys()):
            if not key.startswith('_'):
                if isinstance(data[key], np.ndarray) and len(data[key].shape) == 2:
                    if data[key].shape[0] > data[key].shape[1]:
                        dataset.views.append(np.array(data[key]))
        dataset.labels = data.get('Y', data.get('gnd', data.get('gt'))).flatten().astype(int)
    
    if dataset.labels.min() == 1:
        dataset.labels -= 1
    
    dataset.n_views = len(dataset.views)
    dataset.n_samples = dataset.views[0].shape[0]
    dataset.n_clusters = len(np.unique(dataset.labels))
    
    return dataset


def load_reuters(data_path='./data'):
    """
    Load Reuters Multi-lingual dataset.
    
    5 views: English, French, German, Spanish, Italian.
    """
    dataset = MultiViewDataset('Reuters')
    
    file_path = os.path.join(data_path, 'Reuters.mat')
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Generating synthetic data.")
        return _generate_synthetic_multiview(
            n_samples=18758, n_clusters=6, n_views=5,
            view_dims=[2000, 2000, 2000, 2000, 2000], name='Reuters'
        )
    
    data = loadmat(file_path)
    
    if 'X' in data:
        views = data['X'].flatten()
        dataset.views = [np.array(v) for v in views]
        dataset.labels = data['Y'].flatten().astype(int)
    
    if dataset.labels.min() == 1:
        dataset.labels -= 1
    
    dataset.n_views = len(dataset.views)
    dataset.n_samples = dataset.views[0].shape[0]
    dataset.n_clusters = len(np.unique(dataset.labels))
    
    return dataset


def load_bbcsport(data_path='./data'):
    """
    Load BBC Sport dataset.
    
    2 views: text from 2 segments.
    544 samples, 5 sports categories.
    """
    dataset = MultiViewDataset('BBCSport')
    
    file_path = os.path.join(data_path, 'bbcsport.mat')
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Generating synthetic data.")
        return _generate_synthetic_multiview(
            n_samples=544, n_clusters=5, n_views=2,
            view_dims=[3183, 3203], name='BBCSport'
        )
    
    data = loadmat(file_path)
    
    if 'X' in data:
        views = data['X'].flatten()
        dataset.views = [np.array(v.toarray() if hasattr(v, 'toarray') else v) 
                        for v in views]
        dataset.labels = data['Y'].flatten().astype(int)
    
    if dataset.labels.min() == 1:
        dataset.labels -= 1
    
    dataset.n_views = len(dataset.views)
    dataset.n_samples = dataset.views[0].shape[0]
    dataset.n_clusters = len(np.unique(dataset.labels))
    
    return dataset


def load_msrcv1(data_path='./data'):
    """
    Load MSRC-v1 dataset.
    
    6 views: CM, GIST, HOG, LBP, SIFT, CENT.
    210 samples, 7 classes.
    """
    dataset = MultiViewDataset('MSRC-v1')
    
    file_path = os.path.join(data_path, 'MSRCV1.mat')
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Generating synthetic data.")
        return _generate_synthetic_multiview(
            n_samples=210, n_clusters=7, n_views=6,
            view_dims=[24, 512, 576, 256, 210, 254], name='MSRC-v1'
        )
    
    data = loadmat(file_path)
    
    if 'X' in data:
        views = data['X'].flatten()
        dataset.views = [np.array(v) for v in views]
        dataset.labels = data['Y'].flatten().astype(int)
    
    if dataset.labels.min() == 1:
        dataset.labels -= 1
    
    dataset.n_views = len(dataset.views)
    dataset.n_samples = dataset.views[0].shape[0]
    dataset.n_clusters = len(np.unique(dataset.labels))
    
    return dataset


def load_nuswide(data_path='./data'):
    """
    Load NUS-WIDE-Object dataset (subset).
    
    5 views: Color Histogram, Color Moments, Color Correlation, 
             Edge Direction Histogram, Wavelet Texture.
    """
    dataset = MultiViewDataset('NUS-WIDE')
    
    file_path = os.path.join(data_path, 'NUSWide.mat')
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Generating synthetic data.")
        return _generate_synthetic_multiview(
            n_samples=30000, n_clusters=31, n_views=5,
            view_dims=[64, 225, 144, 73, 128], name='NUS-WIDE'
        )
    
    data = loadmat(file_path)
    
    if 'X' in data:
        views = data['X'].flatten()
        dataset.views = [np.array(v) for v in views]
        dataset.labels = data['Y'].flatten().astype(int)
    
    if dataset.labels.min() == 1:
        dataset.labels -= 1
    
    dataset.n_views = len(dataset.views)
    dataset.n_samples = dataset.views[0].shape[0]
    dataset.n_clusters = len(np.unique(dataset.labels))
    
    return dataset


def _generate_synthetic_multiview(n_samples, n_clusters, n_views, view_dims, 
                                   name='Synthetic', random_state=42):
    """
    Generate synthetic multi-view data for testing.
    
    Creates more realistic multi-view data with:
    - View-specific cluster structures (some views are more informative)
    - Non-linear transformations
    - Noise and outliers
    """
    np.random.seed(random_state)
    
    dataset = MultiViewDataset(f'{name} (Synthetic)')
    
    # Generate labels (balanced classes)
    samples_per_cluster = n_samples // n_clusters
    labels = np.repeat(np.arange(n_clusters), samples_per_cluster)
    remaining = n_samples - len(labels)
    if remaining > 0:
        labels = np.concatenate([labels, np.random.choice(n_clusters, remaining)])
    
    # Shuffle labels to avoid ordered data
    shuffle_idx = np.random.permutation(n_samples)
    labels = labels[shuffle_idx]
    
    dataset.labels = labels
    
    # Generate shared latent representation
    latent_dim = min(30, min(view_dims) // 2)
    
    # Cluster centers with good separation
    cluster_centers = []
    for k in range(n_clusters):
        # Use orthogonal-ish directions for cluster centers
        center = np.zeros(latent_dim)
        center[k % latent_dim] = 5.0  # Main direction
        center[(k + 1) % latent_dim] = 3.0  # Secondary direction
        center += np.random.randn(latent_dim) * 0.5  # Small random offset
        cluster_centers.append(center)
    cluster_centers = np.array(cluster_centers)
    
    # Generate samples with varying intra-cluster variance
    latent = np.zeros((n_samples, latent_dim))
    for k in range(n_clusters):
        mask = labels == k
        n_k = mask.sum()
        # Add some structure within clusters
        intra_cluster_std = 0.8 + 0.4 * np.random.rand()
        latent[mask] = cluster_centers[k] + np.random.randn(n_k, latent_dim) * intra_cluster_std
    
    # Generate views with different characteristics
    for v, dim in enumerate(view_dims):
        view_data = np.zeros((n_samples, dim))
        
        # Each view has different "quality" (how well it preserves cluster structure)
        view_quality = 0.5 + 0.5 * np.random.rand()  # 0.5 to 1.0
        
        # Random non-linear projection
        if dim >= latent_dim:
            # Project to higher dimension
            projection1 = np.random.randn(latent_dim, dim) / np.sqrt(latent_dim)
            projection2 = np.random.randn(latent_dim, dim) / np.sqrt(latent_dim)
            
            # Non-linear combination
            linear_part = latent @ projection1
            nonlinear_part = np.tanh(latent @ projection2) * 2
            
            view_data = view_quality * linear_part + (1 - view_quality) * nonlinear_part
        else:
            # Project to lower dimension (information loss)
            projection = np.random.randn(latent_dim, dim) / np.sqrt(latent_dim)
            view_data = latent @ projection
        
        # Add view-specific noise
        noise_level = 0.3 + 0.5 * np.random.rand()  # Different noise per view
        view_data += np.random.randn(n_samples, dim) * noise_level
        
        # Add some outliers (5% of samples)
        n_outliers = int(0.05 * n_samples)
        outlier_idx = np.random.choice(n_samples, n_outliers, replace=False)
        view_data[outlier_idx] += np.random.randn(n_outliers, dim) * 3
        
        # Normalize
        view_data = (view_data - view_data.mean(axis=0)) / (view_data.std(axis=0) + 1e-10)
        
        dataset.views.append(view_data)
    
    dataset.n_views = n_views
    dataset.n_samples = n_samples
    dataset.n_clusters = n_clusters
    
    return dataset


def load_dataset(name, data_path='./data', **kwargs):
    """
    Load dataset by name.
    
    Parameters
    ----------
    name : str
        Dataset name. Options: 'handwritten', 'caltech101-7', 'caltech101-20',
        'scene15', 'reuters', 'bbcsport', 'msrcv1', 'nuswide'
    data_path : str
        Path to data directory.
        
    Returns
    -------
    dataset : MultiViewDataset
        Loaded dataset.
    """
    name_lower = name.lower().replace('-', '').replace('_', '')
    
    loaders = {
        'handwritten': load_handwritten,
        'caltech1017': lambda p: load_caltech101(p, 7),
        'caltech10120': lambda p: load_caltech101(p, 20),
        'caltech101': lambda p: load_caltech101(p, 7),
        'scene15': load_scene15,
        'reuters': load_reuters,
        'bbcsport': load_bbcsport,
        'msrcv1': load_msrcv1,
        'nuswide': load_nuswide,
    }
    
    if name_lower in loaders:
        return loaders[name_lower](data_path)
    else:
        available = list(loaders.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")


def get_available_datasets():
    """Get list of available dataset names."""
    return [
        'Handwritten',
        'Caltech101-7',
        'Caltech101-20',
        'Scene15',
        'Reuters',
        'BBCSport',
        'MSRC-v1',
        'NUS-WIDE'
    ]
