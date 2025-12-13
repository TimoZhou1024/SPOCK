"""
Multi-View Dataset Loaders

Supports various benchmark datasets for multi-view clustering evaluation.
"""

import os
import numpy as np
from scipy.io import loadmat, mmread
from scipy.sparse import issparse
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
    # Handle Caltech101 specific format: data['data'] contains the views
    if 'data' in data:
        # 'data' is a (1, 6) cell array containing 6 view matrices
        data_cell = data['data'].flatten()  # Convert to 1D array of objects
        dataset.views = [np.array(v, dtype=np.float32) for v in data_cell]
        dataset.labels = data['Y'].flatten().astype(int)
    elif 'X' in data:
        views = data['X'].flatten()
        dataset.views = [np.array(v, dtype=np.float32) for v in views]
        dataset.labels = data['Y'].flatten().astype(int)
    else:
        for key in sorted(data.keys()):
            if 'fea' in key.lower() or key.startswith('X'):
                dataset.views.append(np.array(data[key], dtype=np.float32))
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
    
    Supports multiple formats:
    1. scene15.mat with 'X' cell array and 'gt' labels
    2. Scene15.mat with 'X' array and 'Y' labels  
    3. Raw images: 15-Scene/ folder with subfolders 00-14
       - Extracts features using: Histogram, Edge, Color statistics
    
    4485 samples, 15 scene categories.
    """
    dataset = MultiViewDataset('Scene15')
    
    # Try multiple mat file names
    mat_files = ['scene15.mat', 'Scene15.mat']
    mat_file_path = None
    for fname in mat_files:
        path = os.path.join(data_path, fname)
        if os.path.exists(path):
            mat_file_path = path
            break
    
    image_dir = os.path.join(data_path, '15-Scene')
    
    # Try loading from .mat file first
    if mat_file_path is not None:
        data = loadmat(mat_file_path)
        
        if 'X' in data:
            X = data['X']
            # Handle cell array format: X is (1, n_views), each cell is (features, samples)
            if X.shape[0] == 1 and X.dtype == object:
                views = []
                for i in range(X.shape[1]):
                    v = X[0, i]
                    # Transpose if features > samples (features, samples) -> (samples, features)
                    if v.shape[0] < v.shape[1]:
                        v = v.T
                    views.append(np.array(v, dtype=np.float64))
                dataset.views = views
            else:
                # Flat array format
                views = X.flatten()
                dataset.views = [np.array(v, dtype=np.float64) for v in views]
            
            # Try different label keys
            if 'gt' in data:
                dataset.labels = data['gt'].flatten().astype(int)
            elif 'Y' in data:
                dataset.labels = data['Y'].flatten().astype(int)
            else:
                raise ValueError("No label field found (tried 'gt', 'Y')")
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
    
    # Try loading from image directory
    if os.path.isdir(image_dir):
        try:
            return _load_scene15_from_images(image_dir)
        except Exception as e:
            print(f"Warning: Failed to load Scene15 from images: {e}")
    
    # Fallback: generate synthetic data
    print(f"Warning: Scene15 data not found. Generating synthetic data.")
    return _generate_synthetic_multiview(
        n_samples=4485, n_clusters=15, n_views=3,
        view_dims=[59, 20, 40], name='Scene15'
    )


def _load_scene15_from_images(image_dir):
    """
    Load Scene-15 from raw image files and extract multi-view features.
    
    Views extracted:
    1. Intensity histogram (256-dim)
    2. Edge histogram using Sobel (64-dim) 
    3. Color statistics (mean, std, skewness per channel) + spatial grid (45-dim)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL/Pillow is required to load images. Install with: pip install Pillow")
    
    dataset = MultiViewDataset('Scene15')
    
    # Find all class folders (00-14)
    class_folders = sorted([d for d in os.listdir(image_dir) 
                           if os.path.isdir(os.path.join(image_dir, d)) and d.isdigit()])
    
    if len(class_folders) == 0:
        raise ValueError(f"No class folders found in {image_dir}")
    
    print(f"Loading Scene15 from images: {len(class_folders)} classes found")
    
    all_features_v1 = []  # Intensity histogram
    all_features_v2 = []  # Edge features
    all_features_v3 = []  # Color/spatial features
    all_labels = []
    
    for class_idx, folder in enumerate(class_folders):
        folder_path = os.path.join(image_dir, folder)
        image_files = sorted([f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        print(f"  Class {class_idx} ({folder}): {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            try:
                img = Image.open(img_path)
                
                # Resize to standard size for consistent features
                img = img.resize((128, 128))
                
                # Convert to numpy array
                img_array = np.array(img)
                
                # Extract View 1: Intensity histogram (grayscale)
                if len(img_array.shape) == 3:
                    gray = np.mean(img_array, axis=2)
                else:
                    gray = img_array.astype(float)
                hist_v1, _ = np.histogram(gray.flatten(), bins=64, range=(0, 256), density=True)
                all_features_v1.append(hist_v1)
                
                # Extract View 2: Edge features (Sobel-like)
                edge_features = _extract_edge_features(gray)
                all_features_v2.append(edge_features)
                
                # Extract View 3: Color/spatial statistics
                color_features = _extract_color_spatial_features(img_array)
                all_features_v3.append(color_features)
                
                all_labels.append(class_idx)
                
            except Exception as e:
                print(f"    Warning: Failed to process {img_file}: {e}")
                continue
    
    if len(all_labels) == 0:
        raise ValueError("No images were successfully loaded")
    
    dataset.views = [
        np.array(all_features_v1, dtype=np.float64),
        np.array(all_features_v2, dtype=np.float64),
        np.array(all_features_v3, dtype=np.float64),
    ]
    dataset.labels = np.array(all_labels, dtype=int)
    dataset.n_views = 3
    dataset.n_samples = len(all_labels)
    dataset.n_clusters = len(class_folders)
    
    print(f"Loaded Scene15: {dataset.n_samples} samples, {dataset.n_views} views, "
          f"{dataset.n_clusters} clusters, dims={[v.shape[1] for v in dataset.views]}")
    
    return dataset


def _extract_edge_features(gray_img):
    """Extract edge histogram features using simple gradient operators."""
    # Simple Sobel-like kernels
    # Horizontal gradient
    gx = np.zeros_like(gray_img, dtype=float)
    gx[:, 1:-1] = gray_img[:, 2:] - gray_img[:, :-2]
    
    # Vertical gradient
    gy = np.zeros_like(gray_img, dtype=float)
    gy[1:-1, :] = gray_img[2:, :] - gray_img[:-2, :]
    
    # Gradient magnitude and direction
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    
    # Edge histogram: 8 direction bins x 4x4 spatial grid = 128 dims
    n_dir_bins = 8
    n_spatial = 4
    
    h, w = gray_img.shape
    cell_h, cell_w = h // n_spatial, w // n_spatial
    
    features = []
    for i in range(n_spatial):
        for j in range(n_spatial):
            cell_mag = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cell_dir = direction[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            
            # Histogram of gradient directions weighted by magnitude
            hist, _ = np.histogram(cell_dir.flatten(), bins=n_dir_bins, 
                                  range=(-np.pi, np.pi), weights=cell_mag.flatten())
            # Normalize
            hist = hist / (np.sum(hist) + 1e-10)
            features.extend(hist)
    
    return np.array(features)


def _extract_color_spatial_features(img_array):
    """Extract color and spatial statistics features."""
    features = []
    
    # Ensure 3 channels
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=2)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Global color statistics (mean, std for each channel)
    for c in range(3):
        channel = img_array[:, :, c].astype(float)
        features.append(np.mean(channel) / 255.0)
        features.append(np.std(channel) / 255.0)
    
    # Spatial grid statistics (3x3 grid, mean intensity per cell)
    h, w = img_array.shape[:2]
    gray = np.mean(img_array, axis=2)
    
    n_grid = 3
    cell_h, cell_w = h // n_grid, w // n_grid
    
    for i in range(n_grid):
        for j in range(n_grid):
            cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            features.append(np.mean(cell) / 255.0)
            features.append(np.std(cell) / 255.0)
    
    # Color histogram (simplified: 8 bins per channel)
    for c in range(3):
        hist, _ = np.histogram(img_array[:, :, c].flatten(), bins=8, range=(0, 256), density=True)
        features.extend(hist)
    
    return np.array(features)


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
    Load BBC Sport dataset from MatrixMarket files.
    
    2 views: text from 2 segments (seg1of2, seg2of2).
    544 samples, 5 sports categories (athletics, cricket, football, rugby, tennis).
    
    Data format:
    - bbcsport_seg1of2.mtx, bbcsport_seg2of2.mtx: term-document sparse matrices
    - bbcsport_seg1of2.docs, bbcsport_seg2of2.docs: document names
    - bbcsport.clist: class labels (category: doc1, doc2, ...)
    """
    dataset = MultiViewDataset('BBCSport')
    
    bbcsport_dir = os.path.join(data_path, 'bbcsport')
    mat_file_path = os.path.join(data_path, 'bbcsport.mat')
    
    # Try loading from MatrixMarket files first
    if os.path.isdir(bbcsport_dir):
        try:
            # Load the two views (seg1of2 and seg2of2)
            view1_mtx = os.path.join(bbcsport_dir, 'bbcsport_seg1of2.mtx')
            view2_mtx = os.path.join(bbcsport_dir, 'bbcsport_seg2of2.mtx')
            view1_docs = os.path.join(bbcsport_dir, 'bbcsport_seg1of2.docs')
            view2_docs = os.path.join(bbcsport_dir, 'bbcsport_seg2of2.docs')
            clist_file = os.path.join(bbcsport_dir, 'bbcsport.clist')
            
            if all(os.path.exists(f) for f in [view1_mtx, view2_mtx, view1_docs, view2_docs, clist_file]):
                # Read sparse matrices (term x document) and transpose to (document x term)
                X1 = mmread(view1_mtx).T.tocsr()  # (n_docs, n_terms)
                X2 = mmread(view2_mtx).T.tocsr()
                
                # Read document names for each view
                with open(view1_docs, 'r') as f:
                    docs1 = [line.strip() for line in f if line.strip()]
                with open(view2_docs, 'r') as f:
                    docs2 = [line.strip() for line in f if line.strip()]
                
                # Parse class labels from clist file
                # Format: "category: doc1,doc2,doc3,..."
                doc_to_label = {}
                label_names = []
                with open(clist_file, 'r') as f:
                    for label_idx, line in enumerate(f):
                        line = line.strip()
                        if ':' in line:
                            category, doc_list = line.split(':', 1)
                            category = category.strip()
                            label_names.append(category)
                            for doc in doc_list.split(','):
                                doc = doc.strip()
                                if doc:
                                    doc_to_label[doc] = label_idx
                
                # Find common documents between views
                common_docs = sorted(set(docs1) & set(docs2))
                
                if len(common_docs) == 0:
                    raise ValueError("No common documents between views")
                
                # Create index mappings
                docs1_idx = {doc: i for i, doc in enumerate(docs1)}
                docs2_idx = {doc: i for i, doc in enumerate(docs2)}
                
                # Extract aligned samples
                idx1 = [docs1_idx[doc] for doc in common_docs]
                idx2 = [docs2_idx[doc] for doc in common_docs]
                labels = np.array([doc_to_label.get(doc, -1) for doc in common_docs])
                
                # Filter out documents without labels
                valid_mask = labels >= 0
                idx1 = [idx1[i] for i in range(len(idx1)) if valid_mask[i]]
                idx2 = [idx2[i] for i in range(len(idx2)) if valid_mask[i]]
                labels = labels[valid_mask]
                
                # Extract view matrices
                X1_aligned = X1[idx1].toarray() if issparse(X1) else X1[idx1]
                X2_aligned = X2[idx2].toarray() if issparse(X2) else X2[idx2]
                
                dataset.views = [X1_aligned.astype(np.float64), X2_aligned.astype(np.float64)]
                dataset.labels = labels.astype(int)
                dataset.n_views = 2
                dataset.n_samples = len(labels)
                dataset.n_clusters = len(label_names)
                
                print(f"Loaded BBCSport: {dataset.n_samples} samples, {dataset.n_views} views, "
                      f"{dataset.n_clusters} clusters, dims={[v.shape[1] for v in dataset.views]}")
                
                return dataset
                
        except Exception as e:
            print(f"Warning: Failed to load BBCSport from MatrixMarket files: {e}")
    
    # Fallback: try .mat file
    if os.path.exists(mat_file_path):
        data = loadmat(mat_file_path)
        
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
    
    # Final fallback: generate synthetic data
    print(f"Warning: BBCSport data not found in {bbcsport_dir} or {mat_file_path}. Generating synthetic data.")
    return _generate_synthetic_multiview(
        n_samples=544, n_clusters=5, n_views=2,
        view_dims=[3183, 3203], name='BBCSport'
    )


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
    
    file_path = os.path.join(data_path, 'NUSWIDE.mat')
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Generating synthetic data.")
        return _generate_synthetic_multiview(
            n_samples=30000, n_clusters=31, n_views=5,
            view_dims=[64, 225, 144, 73, 128], name='NUS-WIDE'
        )
    
    data = loadmat(file_path)
    
    # NUSwide format: 'fea' contains views, 'gt' contains labels
    if 'fea' in data and 'gt' in data:
        fea_cell = data['fea'].flatten()  # (1, 5) cell array
        dataset.views = [np.array(v, dtype=np.float32) for v in fea_cell]
        dataset.labels = data['gt'].flatten().astype(int)
    elif 'X' in data:
        views = data['X'].flatten()
        dataset.views = [np.array(v, dtype=np.float32) for v in views]
        dataset.labels = data['Y'].flatten().astype(int)
    else:
        raise ValueError(f"Unexpected data format in {file_path}. Expected 'fea'/'gt' or 'X'/'Y' keys.")
    
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
