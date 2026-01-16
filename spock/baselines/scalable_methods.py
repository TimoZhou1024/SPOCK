"""
Scalable Multi-View Clustering Methods (Near-Linear Complexity)

This module provides implementations of scalable MVC methods that achieve
near-linear O(nm) or O(n log n) complexity, where m << n is the number of anchors.

These methods are directly comparable to SPOCK in terms of scalability and
can handle large-scale datasets efficiently.

Methods included:
    Built-in (pure Python/NumPy):
    - LMVSC: Large-scale Multi-View Spectral Clustering via Bipartite Graph (AAAI 2020)
    - SMVSC: Scalable Multi-View Subspace Clustering with Unified Anchors (ACM MM 2021)
    - FMCNOF: Fast Multi-view Clustering via Nonnegative and Orthogonal Factorization (TIP 2021)
    - EOMSC-CA: Efficient One-pass Multi-view Subspace Clustering with Consensus Anchors (AAAI 2022)
    - BMVC: Binary Multi-View Clustering (TPAMI 2019)
    - FastMVC: Fast Multi-View Clustering (late fusion)

    External (require git clone, PyTorch):
    - SCMVC: Self-Weighted Contrastive Fusion for Deep MVC (IEEE TMM 2024)

All methods follow the unified interface:
    model = MethodWrapper(n_clusters, ...)
    labels = model.fit_predict(X_views)

Complexity comparison:
    - Traditional spectral: O(n^3) or O(n^2 k) for sparse
    - Anchor-based methods: O(nm^2 + m^3) where m << n
    - SPOCK: O(n log n) with OT-based approach

External methods setup:
    1. Clone repository to spock/baselines/external/<method_name>/
    2. Install dependencies (pip install torch)
    3. The wrapper will automatically detect and use the external code
"""

import os
import numpy as np
import warnings
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh, svds
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# Get the external directory path
EXTERNAL_DIR = os.path.join(os.path.dirname(__file__), 'external')


class BaseScalableMVC(ABC):
    """
    Base class for scalable multi-view clustering methods.

    All scalable MVC methods should inherit from this class and implement
    the fit_predict method.

    Attributes
    ----------
    n_clusters : int
        Number of clusters.
    n_anchors : int
        Number of anchor points (controls complexity vs accuracy trade-off).
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, n_clusters, n_anchors=500, random_state=None):
        self.n_clusters = n_clusters
        self.n_anchors = n_anchors
        self.random_state = random_state
        self.labels_ = None
        self.name = "BaseScalableMVC"

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

    def _get_anchors_kmeans(self, X, n_anchors):
        """
        Select anchors using K-Means clustering.

        Complexity: O(n * n_anchors * n_iter)
        """
        n_anchors = min(n_anchors, X.shape[0] // 2)
        kmeans = KMeans(
            n_clusters=n_anchors,
            n_init=1,
            max_iter=50,
            random_state=self.random_state
        )
        kmeans.fit(X)
        return kmeans.cluster_centers_

    def _get_anchors_random(self, X, n_anchors):
        """
        Select anchors via random sampling.

        Complexity: O(n_anchors)
        """
        n_anchors = min(n_anchors, X.shape[0])
        indices = np.random.choice(X.shape[0], n_anchors, replace=False)
        return X[indices].copy()

    def _build_anchor_graph(self, X, anchors, k_neighbors=5):
        """
        Build sparse anchor graph Z where Z[i,j] = similarity between sample i and anchor j.

        Uses k-nearest anchors for each sample to maintain sparsity.

        Complexity: O(n * m) where m = number of anchors

        Parameters
        ----------
        X : np.ndarray
            Data matrix (n_samples, n_features).
        anchors : np.ndarray
            Anchor matrix (n_anchors, n_features).
        k_neighbors : int
            Number of nearest anchors to connect.

        Returns
        -------
        Z : np.ndarray
            Anchor graph matrix (n_samples, n_anchors).
        """
        n_samples = X.shape[0]
        n_anchors = anchors.shape[0]
        k = min(k_neighbors, n_anchors)

        # Compute distances to all anchors
        # Use chunked computation for large datasets
        chunk_size = 5000
        Z = np.zeros((n_samples, n_anchors))

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            X_chunk = X[start:end]

            # Squared Euclidean distance
            distances = cdist(X_chunk, anchors, metric='sqeuclidean')

            # For each sample, find k nearest anchors
            for i, dist_row in enumerate(distances):
                idx = start + i
                nearest = np.argpartition(dist_row, k)[:k]

                # Gaussian kernel weights
                sigma = np.sqrt(dist_row[nearest[-1]]) + 1e-10
                weights = np.exp(-dist_row[nearest] / (2 * sigma ** 2))
                weights = weights / (weights.sum() + 1e-10)

                Z[idx, nearest] = weights

        return Z


class LMVSCWrapper(BaseScalableMVC):
    """
    Large-scale Multi-View Spectral Clustering via Bipartite Graph (LMVSC).

    Paper: "Large-scale Multi-View Spectral Clustering via Bipartite Graph"
    Venue: AAAI 2020
    Authors: Yeqing Li, Feiping Nie, Heng Huang, Junzhou Huang

    Complexity: O(nm^2 + m^3) where m << n

    Key idea:
    - Build bipartite graph between samples and anchors for each view
    - Fuse anchor graphs across views
    - Perform spectral clustering on the small m x m graph
    - Transfer labels back to original samples

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    n_anchors : int, default=500
        Number of anchor points.
    k_neighbors : int, default=5
        Number of nearest anchors for graph construction.
    random_state : int, optional
        Random seed.
    """

    def __init__(self, n_clusters, n_anchors=500, k_neighbors=5, random_state=None):
        super().__init__(n_clusters, n_anchors, random_state)
        self.k_neighbors = k_neighbors
        self.name = 'LMVSC'

    def fit_predict(self, X_views):
        """
        Fit LMVSC and return cluster labels.

        Algorithm:
        1. For each view, select anchors and build anchor graph Z_v
        2. Compute consensus anchor graph: Z = mean(Z_v)
        3. Construct small bipartite Laplacian on anchors
        4. Spectral embedding of anchors -> transfer to samples
        """
        self._set_seed()

        n_samples = X_views[0].shape[0]
        n_views = len(X_views)
        n_anchors = min(self.n_anchors, n_samples // 2)

        # Step 1: Build anchor graphs for each view
        Z_list = []
        for v, X_v in enumerate(X_views):
            X_norm = normalize(X_v, norm='l2')
            anchors = self._get_anchors_kmeans(X_norm, n_anchors)
            Z_v = self._build_anchor_graph(X_norm, anchors, self.k_neighbors)
            Z_list.append(Z_v)

        # Step 2: Fuse anchor graphs (simple average)
        Z = np.mean(Z_list, axis=0)

        # Step 3: Build anchor similarity matrix S = Z^T * Z (m x m)
        # This is the key to scalability: we work with m x m matrix instead of n x n
        S = Z.T @ Z

        # Normalize: D^{-1/2} S D^{-1/2}
        d = np.sqrt(S.sum(axis=1) + 1e-10)
        D_inv_sqrt = np.diag(1.0 / d)
        S_norm = D_inv_sqrt @ S @ D_inv_sqrt

        # Step 4: Eigen decomposition of m x m matrix
        # Get top k eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(S_norm)
            # Take top k (largest eigenvalues)
            idx = np.argsort(eigenvalues)[::-1][:self.n_clusters]
            U_anchors = eigenvectors[:, idx]
        except:
            # Fallback to simple K-Means
            X_concat = np.hstack([normalize(x) for x in X_views])
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
            return kmeans.fit_predict(X_concat)

        # Step 5: Transfer embedding from anchors to samples
        # U_samples = Z @ U_anchors (n x k)
        U_samples = Z @ U_anchors
        U_samples = normalize(U_samples, norm='l2', axis=1)

        # Step 6: K-Means on embedding
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state
        )
        return kmeans.fit_predict(U_samples)


class SMVSCWrapper(BaseScalableMVC):
    """
    Scalable Multi-View Subspace Clustering with Unified Anchors (SMVSC).

    Paper: "Scalable Multi-view Subspace Clustering with Unified Anchors"
    Venue: ACM Multimedia 2021 / TKDE 2022
    Authors: Mengjing Sun, Pei Zhang, Siwei Wang, et al.
    GitHub: https://github.com/ManshengChen/Code-for-SMVSC-TKDE

    Complexity: O(nm^2 + m^3)

    Key idea:
    - Learn unified anchors shared across all views
    - Jointly optimize anchor selection and graph construction
    - Self-representation on anchor level

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    n_anchors : int, default=500
        Number of anchor points.
    k_neighbors : int, default=7
        Number of nearest anchors.
    lambda_reg : float, default=0.1
        Regularization weight.
    n_iter : int, default=10
        Number of optimization iterations.
    random_state : int, optional
        Random seed.
    """

    def __init__(self, n_clusters, n_anchors=500, k_neighbors=7,
                 lambda_reg=0.1, n_iter=10, random_state=None):
        super().__init__(n_clusters, n_anchors, random_state)
        self.k_neighbors = k_neighbors
        self.lambda_reg = lambda_reg
        self.n_iter = n_iter
        self.name = 'SMVSC'

    def fit_predict(self, X_views):
        """
        Fit SMVSC and return cluster labels.

        Algorithm:
        1. Initialize unified anchors from concatenated features
        2. Iteratively update:
           a. Anchor graphs Z_v for each view
           b. Unified anchors A
           c. Self-representation matrix C
        3. Spectral clustering on final consensus
        """
        self._set_seed()

        n_samples = X_views[0].shape[0]
        n_views = len(X_views)
        n_anchors = min(self.n_anchors, n_samples // 2)

        # Normalize all views
        X_norm_list = [normalize(X_v, norm='l2') for X_v in X_views]

        # Initialize unified anchors from concatenated normalized views
        X_concat = np.hstack(X_norm_list)
        anchors_unified = self._get_anchors_kmeans(X_concat, n_anchors)

        # Split unified anchors back to per-view anchors
        dims = [X_v.shape[1] for X_v in X_views]
        anchors_per_view = []
        offset = 0
        for d in dims:
            anchors_per_view.append(anchors_unified[:, offset:offset+d])
            offset += d

        # Build initial anchor graphs
        Z_list = []
        for v in range(n_views):
            Z_v = self._build_anchor_graph(X_norm_list[v], anchors_per_view[v], self.k_neighbors)
            Z_list.append(Z_v)

        # Iterative refinement (simplified version)
        for iteration in range(self.n_iter):
            # Update consensus Z
            Z = np.mean(Z_list, axis=0)

            # Update view-specific graphs with adaptive weights
            view_quality = []
            for v in range(n_views):
                # Measure view quality by graph smoothness
                diff = Z_list[v] - Z
                quality = 1.0 / (np.linalg.norm(diff, 'fro') + 1e-10)
                view_quality.append(quality)

            # Normalize weights
            weights = np.array(view_quality)
            weights = weights / weights.sum()

            # Weighted fusion
            Z = sum(w * Z_v for w, Z_v in zip(weights, Z_list))

        # Anchor-level self-representation: S = Z^T Z
        S = Z.T @ Z

        # Normalize
        d = np.sqrt(S.sum(axis=1) + 1e-10)
        D_inv_sqrt = np.diag(1.0 / d)
        S_norm = D_inv_sqrt @ S @ D_inv_sqrt

        # Spectral embedding
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(S_norm)
            idx = np.argsort(eigenvalues)[::-1][:self.n_clusters]
            U_anchors = eigenvectors[:, idx]
        except:
            # Fallback
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
            return kmeans.fit_predict(X_concat)

        # Transfer to samples
        U_samples = Z @ U_anchors
        U_samples = normalize(U_samples, norm='l2', axis=1)

        # K-Means
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        return kmeans.fit_predict(U_samples)


class FMCNOFWrapper(BaseScalableMVC):
    """
    Fast Multi-view Clustering via Nonnegative and Orthogonal Factorization (FMCNOF).

    Paper: "Fast Multi-view Clustering via Nonnegative and Orthogonal Factorization"
    Venue: IEEE TIP 2021
    Authors: Fangchen Yu, et al.

    Complexity: O(nmk) where k = n_clusters

    Key idea:
    - Use nonnegative matrix factorization with orthogonal constraints
    - Factorize X ≈ W @ H where H gives cluster indicators
    - Shared H across views with view-specific W

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    n_iter : int, default=100
        Number of NMF iterations.
    alpha : float, default=1.0
        Weight for orthogonality constraint.
    random_state : int, optional
        Random seed.
    """

    def __init__(self, n_clusters, n_iter=100, alpha=1.0, random_state=None):
        super().__init__(n_clusters, n_anchors=0, random_state=random_state)
        self.n_iter = n_iter
        self.alpha = alpha
        self.name = 'FMCNOF'

    def fit_predict(self, X_views):
        """
        Fit FMCNOF and return cluster labels.

        Algorithm:
        1. Initialize shared H (n x k) and per-view W_v (d_v x k)
        2. Alternating optimization:
           a. Fix H, update W_v for each view
           b. Fix W_v, update H (consensus)
        3. Cluster assignment from H
        """
        self._set_seed()

        n_samples = X_views[0].shape[0]
        n_views = len(X_views)
        k = self.n_clusters

        # Ensure non-negative (shift if needed)
        X_nn = []
        for X_v in X_views:
            X_shifted = X_v - X_v.min() + 1e-10
            X_nn.append(X_shifted)

        # Initialize H using K-Means on concatenated data
        X_concat = np.hstack([normalize(x) for x in X_views])
        kmeans_init = KMeans(n_clusters=k, n_init=3, random_state=self.random_state)
        init_labels = kmeans_init.fit_predict(X_concat)

        # Convert labels to indicator matrix H (n x k)
        H = np.zeros((n_samples, k))
        for i, label in enumerate(init_labels):
            H[i, label] = 1.0
        H = H + 0.1  # Avoid zeros

        # Initialize W_v for each view
        W_list = []
        for X_v in X_nn:
            # W_v = X_v^T @ H @ (H^T H)^{-1}
            HtH_inv = np.linalg.pinv(H.T @ H + 1e-6 * np.eye(k))
            W_v = X_v.T @ H @ HtH_inv
            W_v = np.maximum(W_v, 1e-10)
            W_list.append(W_v)

        # Alternating optimization
        eps = 1e-10
        for iteration in range(self.n_iter):
            # Update W_v for each view (fix H)
            for v in range(n_views):
                X_v = X_nn[v]
                W_v = W_list[v]

                # Multiplicative update
                numerator = X_v.T @ H
                denominator = W_v @ H.T @ H + eps
                W_list[v] = W_v * (numerator / denominator)

            # Update H (fix W_v)
            numerator = np.zeros((n_samples, k))
            denominator = np.zeros((n_samples, k))

            for v in range(n_views):
                X_v = X_nn[v]
                W_v = W_list[v]
                numerator += X_v @ W_v
                denominator += H @ W_v.T @ W_v

            # Add orthogonality regularization
            denominator += self.alpha * H @ H.T @ H

            H = H * (numerator / (denominator + eps))
            H = np.maximum(H, eps)

        # Get labels from H
        labels = np.argmax(H, axis=1)

        return labels


class EOMSCCAWrapper(BaseScalableMVC):
    """
    Efficient One-pass Multi-view Subspace Clustering with Consensus Anchors (EOMSC-CA).

    Paper: "Efficient One-pass Multi-view Subspace Clustering with Consensus Anchors"
    Venue: AAAI 2022
    Authors: Suyuan Liu, et al.

    Complexity: O(nm^2 + m^3) - true one-pass complexity

    Key idea:
    - Single pass through data to construct anchor graph
    - Consensus anchor learning across views
    - No iterative optimization required

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    n_anchors : int, default=500
        Number of anchor points.
    k_neighbors : int, default=5
        Number of nearest anchors.
    random_state : int, optional
        Random seed.
    """

    def __init__(self, n_clusters, n_anchors=500, k_neighbors=5, random_state=None):
        super().__init__(n_clusters, n_anchors, random_state)
        self.k_neighbors = k_neighbors
        self.name = 'EOMSC-CA'

    def fit_predict(self, X_views):
        """
        Fit EOMSC-CA and return cluster labels.

        Algorithm (one-pass):
        1. Randomly sample initial anchors
        2. Build consensus anchor graph in one pass
        3. Spectral clustering on anchor graph
        """
        self._set_seed()

        n_samples = X_views[0].shape[0]
        n_views = len(X_views)
        n_anchors = min(self.n_anchors, n_samples // 2)

        # Step 1: Sample random anchors (faster than K-Means for one-pass)
        anchor_indices = np.random.choice(n_samples, n_anchors, replace=False)

        # Step 2: Build anchor graphs and fuse in one pass
        Z_consensus = np.zeros((n_samples, n_anchors))

        for v, X_v in enumerate(X_views):
            X_norm = normalize(X_v, norm='l2')
            anchors = X_norm[anchor_indices]

            # Build sparse anchor graph
            Z_v = self._build_anchor_graph(X_norm, anchors, self.k_neighbors)
            Z_consensus += Z_v

        # Average
        Z_consensus /= n_views

        # Step 3: Bipartite spectral clustering
        # Build affinity on anchors: S = Z^T Z
        S = Z_consensus.T @ Z_consensus

        # Symmetric normalization
        d = np.sqrt(np.maximum(S.sum(axis=1), 1e-10))
        D_inv_sqrt = np.diag(1.0 / d)
        S_norm = D_inv_sqrt @ S @ D_inv_sqrt

        # Eigen decomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(S_norm)
            idx = np.argsort(eigenvalues)[::-1][:self.n_clusters]
            U_anchors = eigenvectors[:, idx]
        except:
            # Fallback
            X_concat = np.hstack([normalize(x) for x in X_views])
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
            return kmeans.fit_predict(X_concat)

        # Transfer to samples
        U_samples = Z_consensus @ U_anchors
        U_samples = normalize(U_samples, norm='l2', axis=1)

        # K-Means
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        return kmeans.fit_predict(U_samples)


class BMVCWrapper(BaseScalableMVC):
    """
    Binary Multi-View Clustering (BMVC).

    Paper: "Binary Multi-View Clustering"
    Venue: IEEE TPAMI 2019
    Authors: Zhang, Zheng, Liu, et al.

    Complexity: O(n * m * b) where b = binary code length

    Key idea:
    - Learn binary codes for samples
    - Binary representations enable O(1) Hamming distance computation
    - Extremely efficient for large-scale data

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    n_bits : int, default=64
        Length of binary codes.
    n_iter : int, default=50
        Number of optimization iterations.
    random_state : int, optional
        Random seed.
    """

    def __init__(self, n_clusters, n_bits=64, n_iter=50, random_state=None):
        super().__init__(n_clusters, n_anchors=0, random_state=random_state)
        self.n_bits = n_bits
        self.n_iter = n_iter
        self.name = 'BMVC'

    def fit_predict(self, X_views):
        """
        Fit BMVC and return cluster labels.

        Algorithm:
        1. Learn binary codes B for each sample
        2. Optimize: min ||B - sign(W_v @ X_v)||
        3. Cluster binary codes using Hamming distance K-Means
        """
        self._set_seed()

        n_samples = X_views[0].shape[0]
        n_views = len(X_views)
        b = self.n_bits

        # Normalize views
        X_norm = [normalize(X_v, norm='l2') for X_v in X_views]

        # Initialize binary codes using PCA on concatenated data
        X_concat = np.hstack(X_norm)

        # Use random projection for initialization (faster than PCA for large data)
        W_init = np.random.randn(X_concat.shape[1], b) / np.sqrt(b)
        B = np.sign(X_concat @ W_init)
        B[B == 0] = 1

        # Initialize projection matrices W_v for each view
        W_list = []
        for X_v in X_norm:
            W_v = np.linalg.lstsq(X_v, B, rcond=None)[0]
            W_list.append(W_v)

        # Alternating optimization
        for iteration in range(self.n_iter):
            # Update B (fix W_v)
            B_new = np.zeros((n_samples, b))
            for v in range(n_views):
                B_new += X_norm[v] @ W_list[v]
            B = np.sign(B_new)
            B[B == 0] = 1

            # Update W_v (fix B)
            for v in range(n_views):
                W_list[v] = np.linalg.lstsq(X_norm[v], B, rcond=None)[0]

        # Cluster binary codes
        # Convert to {0,1} for K-Means
        B_01 = (B + 1) / 2

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        return kmeans.fit_predict(B_01)


class FastMVCWrapper(BaseScalableMVC):
    """
    Fast Multi-View Clustering (FastMVC) - Anchor-free Late Fusion.

    A simple but effective baseline that achieves near-linear complexity
    by using efficient per-view clustering and late fusion.

    Complexity: O(n * k * iter) - truly linear in n

    Key idea:
    - Run K-Means on each view independently
    - Fuse cluster assignments using voting/consensus

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    n_init : int, default=3
        Number of K-Means initializations per view.
    random_state : int, optional
        Random seed.
    """

    def __init__(self, n_clusters, n_init=3, random_state=None):
        super().__init__(n_clusters, n_anchors=0, random_state=random_state)
        self.n_init = n_init
        self.name = 'FastMVC'

    def fit_predict(self, X_views):
        """
        Fit FastMVC and return cluster labels.

        Algorithm:
        1. Run K-Means on each view
        2. Build co-association matrix from all clusterings
        3. Final clustering on co-association
        """
        self._set_seed()

        n_samples = X_views[0].shape[0]
        n_views = len(X_views)

        # Step 1: Get cluster assignments from each view
        all_labels = []
        for X_v in X_views:
            X_norm = normalize(X_v, norm='l2')
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                n_init=self.n_init,
                random_state=self.random_state
            )
            labels_v = kmeans.fit_predict(X_norm)
            all_labels.append(labels_v)

        # Step 2: Build co-association matrix (sparse version for scalability)
        # C[i,j] = number of times samples i,j are in same cluster
        # For large n, we use a sampling-based approximation

        if n_samples <= 10000:
            # Exact co-association for smaller datasets
            C = np.zeros((n_samples, n_samples))
            for labels in all_labels:
                for c in range(self.n_clusters):
                    mask = labels == c
                    C[np.ix_(mask, mask)] += 1
            C = C / n_views

            # Spectral clustering on C
            d = np.sqrt(C.sum(axis=1) + 1e-10)
            D_inv_sqrt = np.diag(1.0 / d)
            C_norm = D_inv_sqrt @ C @ D_inv_sqrt

            eigenvalues, eigenvectors = np.linalg.eigh(C_norm)
            idx = np.argsort(eigenvalues)[::-1][:self.n_clusters]
            U = eigenvectors[:, idx]
            U = normalize(U, norm='l2', axis=1)

            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
            return kmeans.fit_predict(U)
        else:
            # For large datasets, use voting-based consensus
            # Convert each clustering to one-hot and concatenate
            indicators = []
            for labels in all_labels:
                H = np.zeros((n_samples, self.n_clusters))
                H[np.arange(n_samples), labels] = 1
                indicators.append(H)

            # Concatenate: (n_samples, n_views * n_clusters)
            H_concat = np.hstack(indicators)

            # Final K-Means on concatenated indicators
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
            return kmeans.fit_predict(H_concat)


# Registry of available scalable methods (built-in)
SCALABLE_METHODS = {
    'LMVSC': LMVSCWrapper,
    'SMVSC': SMVSCWrapper,
    'FMCNOF': FMCNOFWrapper,
    'EOMSC-CA': EOMSCCAWrapper,
    'BMVC': BMVCWrapper,
    'FastMVC': FastMVCWrapper,
}


# =============================================================================
# External Methods (require git clone)
# =============================================================================

class BaseExternalMVC(BaseScalableMVC):
    """
    Base class for external multi-view clustering methods that require git clone.

    Subclasses should implement:
    - _get_external_paths(): returns list of possible directory names
    - _run_external(): runs the external implementation
    - _run_fallback(): fallback when external code is not available
    """

    def __init__(self, n_clusters, device='auto', random_state=None):
        super().__init__(n_clusters, n_anchors=0, random_state=random_state)

        # Auto-detect device for PyTorch-based methods
        if device == 'auto':
            try:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self.device = 'cpu'
        else:
            self.device = device

    def _get_external_paths(self):
        """Return list of possible directory names for this method."""
        raise NotImplementedError

    def _find_external_path(self):
        """Find the external code directory."""
        for dirname in self._get_external_paths():
            path = os.path.join(EXTERNAL_DIR, dirname)
            if os.path.exists(path):
                return path
        return None

    def _run_external(self, X_views):
        """Run the external implementation."""
        raise NotImplementedError

    def _run_fallback(self, X_views):
        """Fallback implementation when external code is not available."""
        # Default fallback: concatenate + K-Means
        X_concat = np.hstack([normalize(x) for x in X_views])
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        return kmeans.fit_predict(X_concat)

    def fit_predict(self, X_views):
        self._set_seed()

        external_path = self._find_external_path()

        if external_path is not None:
            try:
                # Add external path to sys.path temporarily
                import sys
                if external_path not in sys.path:
                    sys.path.insert(0, external_path)

                return self._run_external(X_views)
            except Exception as e:
                warnings.warn(f"{self.name} failed: {e}. Using fallback.")
                return self._run_fallback(X_views)
        else:
            warnings.warn(
                f"{self.name} external code not found. Using fallback. "
                f"To use the original implementation, clone the repository to: "
                f"{EXTERNAL_DIR}/{self._get_external_paths()[0]}/"
            )
            return self._run_fallback(X_views)


class SCMVCWrapper(BaseExternalMVC):
    """
    Self-Weighted Contrastive Fusion for Deep Multi-View Clustering (SCMVC).

    Paper: "Self-Weighted Contrastive Fusion for Deep Multi-View Clustering"
    Venue: IEEE Transactions on Multimedia (TMM) 2024
    GitHub: https://github.com/SongwuJob/SCMVC

    Setup:
        cd spock/baselines/external
        git clone https://github.com/SongwuJob/SCMVC.git

    Requirements:
        - PyTorch >= 1.12.0
        - scikit-learn

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    pre_epochs : int, default=200
        Number of pretraining epochs.
    con_epochs : int, default=50
        Number of contrastive training epochs.
    batch_size : int, default=256
        Batch size for training.
    learning_rate : float, default=0.0003
        Learning rate.
    feature_dim : int, default=64
        Latent feature dimension.
    high_feature_dim : int, default=20
        High-level feature dimension.
    temperature : float, default=1.0
        Temperature for contrastive loss.
    device : str, default='auto'
        Device to use ('cuda', 'cpu', or 'auto').
    random_state : int, optional
        Random seed.
    """

    def __init__(self, n_clusters, pre_epochs=200, con_epochs=50, batch_size=256,
                 learning_rate=0.0003, feature_dim=64, high_feature_dim=20,
                 temperature=1.0, device='auto', random_state=None):
        super().__init__(n_clusters, device, random_state)
        self.pre_epochs = pre_epochs
        self.con_epochs = con_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.feature_dim = feature_dim
        self.high_feature_dim = high_feature_dim
        self.temperature = temperature
        self.name = 'SCMVC'

    def _get_external_paths(self):
        return ['SCMVC', 'scmvc']

    def _run_external(self, X_views):
        """Run the external SCMVC implementation."""
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.cluster import KMeans
        import sys
        import os

        # Add external SCMVC path
        external_path = self._find_external_path()
        if external_path is None:
            raise ImportError("SCMVC external code not found. Please clone the repository.")
        
        if external_path not in sys.path:
            sys.path.insert(0, external_path)

        # Import from external SCMVC
        from network import Network
        from loss import ContrastiveLoss

        device = torch.device(self.device)
        n_views = len(X_views)
        n_samples = X_views[0].shape[0]
        dims = [x.shape[1] for x in X_views]

        # Create dataset (matching SCMVC's expected format)
        class MVDataset(Dataset):
            def __init__(self, views, labels=None):
                self.views = [torch.FloatTensor(v) for v in views]
                self.n_samples = views[0].shape[0]
                self.labels = labels if labels is not None else np.zeros(self.n_samples)

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                return [v[idx] for v in self.views], self.labels[idx], idx

        # Normalize data
        X_normalized = []
        for x in X_views:
            scaler = MinMaxScaler()
            X_normalized.append(scaler.fit_transform(x))

        dataset = MVDataset(X_normalized)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Initialize model
        model = Network(n_views, dims, self.feature_dim, self.high_feature_dim, device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        contrastiveloss = ContrastiveLoss(self.batch_size, self.temperature, device).to(device)
        mse = nn.MSELoss()

        # Helper function for computing view weights
        def compute_view_value(rs, H, view):
            N = H.shape[0]
            w = []
            global_sim = torch.matmul(H, H.t())
            for v in range(view):
                view_sim = torch.matmul(rs[v], rs[v].t())
                related_sim = torch.matmul(rs[v], H.t())
                w_v = (torch.sum(view_sim) + torch.sum(global_sim) - 2 * torch.sum(related_sim)) / (N * N)
                w.append(torch.exp(-w_v))
            w = torch.stack(w)
            w = w / torch.sum(w)
            return w.squeeze()

        # Pretraining phase
        model.train()
        for epoch in range(self.pre_epochs):
            for batch_idx, (xs, _, _) in enumerate(data_loader):
                for v in range(n_views):
                    xs[v] = xs[v].to(device)
                optimizer.zero_grad()
                xrs, zs, rs, H = model(xs)
                loss = sum([mse(xs[v], xrs[v]) for v in range(n_views)])
                loss.backward()
                optimizer.step()

        # Contrastive training phase
        for epoch in range(self.con_epochs):
            for batch_idx, (xs, _, _) in enumerate(data_loader):
                for v in range(n_views):
                    xs[v] = xs[v].to(device)
                optimizer.zero_grad()
                xrs, zs, rs, H = model(xs)

                # Compute adaptive weights
                with torch.no_grad():
                    w = compute_view_value(rs, H, n_views)

                loss_list = []
                for v in range(n_views):
                    loss_list.append(contrastiveloss(H, rs[v], w[v]))
                    loss_list.append(mse(xs[v], xrs[v]))
                loss = sum(loss_list)
                loss.backward()
                optimizer.step()

        # Get predictions using K-means on H
        model.eval()
        full_loader = DataLoader(dataset, batch_size=n_samples, shuffle=False)

        with torch.no_grad():
            for xs, _, _ in full_loader:
                for v in range(n_views):
                    xs[v] = xs[v].to(device)
                xrs, zs, rs, H = model(xs)

        # K-means clustering on global features H
        H_np = H.cpu().data.numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=100, random_state=self.random_state)
        labels = kmeans.fit_predict(H_np)

        return labels

    def _run_fallback(self, X_views):
        """Fallback to simple spectral clustering on concatenated views."""
        from sklearn.cluster import SpectralClustering
        import numpy as np
        
        # Concatenate all views
        X_concat = np.hstack(X_views)
        
        # Use spectral clustering as fallback
        spectral = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='nearest_neighbors',
            n_neighbors=10,
            random_state=self.random_state
        )
        labels = spectral.fit_predict(X_concat)
        
        return labels


class EFIMVCWrapper(BaseExternalMVC):
    """
    Efficient Federated Incomplete Multi-View Clustering (EFIMVC).

    Paper: "Efficient Federated Incomplete Multi-View Clustering"
    Venue: ICML 2025
    GitHub: https://github.com/Tracesource/EFIMVC

    Note: Original implementation is in MATLAB. This wrapper provides a
    Python approximation based on the paper's methodology.

    Setup:
        cd spock/baselines/external
        git clone https://github.com/Tracesource/EFIMVC.git

    The algorithm uses:
    1. Client-side pre-training: Learn local anchor representations via constrained optimization
    2. Server aggregation: Fuse client representations 
    3. Iterative refinement with orthogonal projections
    4. Final clustering via SVD on aggregated representation

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    n_anchors : int, default=None
        Number of anchors (lambda1 in paper). If None, uses 2*n_clusters.
    lambda2 : float, default=0.1
        Regularization for client optimization.
    lambda3 : float, default=0.1
        Regularization for server optimization.
    max_iter : int, default=5
        Maximum iterations for alternating optimization.
    random_state : int, optional
        Random seed.
    """

    def __init__(self, n_clusters, n_anchors=None, lambda2=0.1, lambda3=0.1,
                 max_iter=5, device='auto', random_state=None):
        super().__init__(n_clusters, device, random_state)
        self.n_anchors = n_anchors if n_anchors is not None else 2 * n_clusters
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.max_iter = max_iter
        self.name = 'EFIMVC'

    def _get_external_paths(self):
        return ['EFIMVC', 'efimvc']

    def _run_external(self, X_views):
        """
        EFIMVC is implemented in MATLAB. Check if MATLAB engine is available,
        otherwise fall back to Python implementation.
        """
        try:
            import matlab.engine
            return self._run_matlab(X_views)
        except ImportError:
            # MATLAB Engine not available, use Python approximation
            return self._run_python(X_views)

    def _run_matlab(self, X_views):
        """Run original MATLAB implementation via MATLAB Engine."""
        import matlab.engine
        import matlab
        
        eng = matlab.engine.start_matlab()
        
        # Add EFIMVC path
        external_path = self._find_external_path()
        eng.addpath(eng.genpath(external_path))
        
        # Convert data to MATLAB format
        n_views = len(X_views)
        n_samples = X_views[0].shape[0]
        
        # Prepare data cell array
        X_matlab = []
        for v in range(n_views):
            X_matlab.append(matlab.double(X_views[v].T.tolist()))  # MATLAB expects features x samples
        
        # Create indicator matrix (all samples complete for standard MVC)
        ind = matlab.double([[1.0] * n_samples for _ in range(n_views)])
        ind = eng.transpose(ind)
        
        # Run EFIMVC core algorithm
        lambda1 = float(self.n_anchors)
        lambda2 = float(self.lambda2)
        lambda3 = float(self.lambda3)
        
        # Client pre-training
        Z_list = []
        for v in range(n_views):
            Z = eng.client_pretrain(X_matlab[v], lambda1, lambda2, 
                                    eng.transpose(matlab.double([ind[i][v] for i in range(n_samples)])))
            Z_list.append(Z)
        
        # Server pre-training
        Zall = eng.server_pretrain(Z_list, ind, lambda1)
        
        # Iterative refinement
        P_list = [matlab.double(np.eye(int(lambda1)).tolist()) for _ in range(n_views)]
        
        for _ in range(self.max_iter):
            # Client training
            S_list = []
            for v in range(n_views):
                Z_new, S, _ = eng.client_train(X_matlab[v], Z_list[v], lambda1, lambda2,
                                                eng.transpose(matlab.double([ind[i][v] for i in range(n_samples)])),
                                                Zall, P_list[v], nargout=3)
                Z_list[v] = Z_new
                S_list.append(S)
            
            # Server training
            Zall, P_list, _ = eng.server_train(P_list, Zall, Z_list, S_list, lambda3, ind, nargout=3)
        
        # Final clustering via SVD
        Zall_np = np.array(eng.transpose(Zall))
        U, _, _ = np.linalg.svd(Zall_np, full_matrices=False)
        U = U[:, :self.n_clusters]
        
        # K-means on SVD embedding
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(U)
        
        eng.quit()
        return labels

    def _run_python(self, X_views):
        """
        Python approximation of EFIMVC algorithm.
        
        Core idea: Learn anchor-based representations per view, aggregate them,
        and perform spectral clustering on the fused representation.
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
        
        n_views = len(X_views)
        n_samples = X_views[0].shape[0]
        n_anchors = min(self.n_anchors, n_samples // 2)
        
        # Step 1: Client-side anchor learning (per view)
        Z_views = []
        A_views = []  # Anchor matrices
        
        for v in range(n_views):
            X = X_views[v]
            
            # Select anchors via K-means
            kmeans = KMeans(n_clusters=n_anchors, random_state=self.random_state, n_init=3, max_iter=50)
            kmeans.fit(X)
            A = kmeans.cluster_centers_  # (n_anchors, d)
            A_views.append(A)
            
            # Compute soft assignment matrix Z: each sample's representation in anchor space
            # Z_ij = similarity between sample i and anchor j, normalized
            # Using RBF kernel similarity
            from scipy.spatial.distance import cdist
            D = cdist(X, A, 'euclidean')
            sigma = np.median(D) + 1e-10
            S = np.exp(-D**2 / (2 * sigma**2))
            
            # Row-normalize to get soft assignment (simplex constraint approximation)
            Z = S / (S.sum(axis=1, keepdims=True) + 1e-10)
            Z_views.append(Z)  # (n_samples, n_anchors)
        
        # Step 2: Server-side aggregation with orthogonal projection
        # Initialize aggregated representation
        Zall = np.mean(Z_views, axis=0)  # Simple average fusion
        
        # Iterative refinement with orthogonal projections
        P_views = [np.eye(n_anchors) for _ in range(n_views)]
        
        for iteration in range(self.max_iter):
            # Update per-view representations with projection
            Z_projected = []
            for v in range(n_views):
                # Project view representation towards consensus
                Z_proj = Z_views[v] @ P_views[v]
                Z_projected.append(Z_proj)
            
            # Update consensus (server aggregation)
            Zall_new = np.mean(Z_projected, axis=0)
            
            # Update orthogonal projections (Procrustes-like)
            for v in range(n_views):
                # Find orthogonal P that aligns Z_v with Zall
                M = Z_views[v].T @ Zall_new
                U, _, Vt = np.linalg.svd(M, full_matrices=False)
                P_views[v] = U @ Vt
            
            # Check convergence
            diff = np.linalg.norm(Zall_new - Zall) / (np.linalg.norm(Zall) + 1e-10)
            Zall = Zall_new
            if diff < 1e-4:
                break
        
        # Step 3: Spectral clustering on aggregated representation
        # SVD for dimensionality reduction
        U, s, Vt = np.linalg.svd(Zall, full_matrices=False)
        
        # Use top-k singular vectors
        k = min(self.n_clusters, U.shape[1])
        embedding = U[:, :k] * s[:k]
        
        # Normalize embedding
        embedding = normalize(embedding, axis=1)
        
        # Final K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(embedding)
        
        return labels

    def _run_fallback(self, X_views):
        """Fallback to Python implementation."""
        return self._run_python(X_views)


class ALPCWrapper(BaseExternalMVC):
    """
    Anchor Learning with Potential Cluster Constraints for Multi-view Clustering (ALPC).

    Paper: "Anchor Learning with Potential Cluster Constraints for Multi-view Clustering"
    Venue: AAAI 2025
    GitHub: https://github.com/whbdmu/ALPC

    Note: Original implementation is in MATLAB. This wrapper provides a
    Python implementation based on the paper's methodology.

    The algorithm:
    1. Learns view-specific anchor matrices A_v
    2. Imposes shared potential clustering semantic constraints
    3. Aligns clustering centers of data with clustering centers of anchors
    4. Uses orthogonal constraints for discriminative anchors

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    n_anchors_per_cluster : int, default=2
        Number of anchors per cluster (m in paper). Total anchors = m * k.
    alpha : float, default=1.0
        Weight for orthogonal constraint.
    beta : float, default=1.0
        Weight for cluster constraint.
    max_iter : int, default=50
        Maximum iterations.
    random_state : int, optional
        Random seed.
    """

    def __init__(self, n_clusters, n_anchors_per_cluster=2, alpha=1.0, beta=1.0,
                 max_iter=50, device='auto', random_state=None):
        super().__init__(n_clusters, device, random_state)
        self.n_anchors_per_cluster = n_anchors_per_cluster
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.name = 'ALPC'

    def _get_external_paths(self):
        return ['ALPC', 'alpc', 'ALPC/ALPC']

    def _run_external(self, X_views):
        """Run Python implementation of ALPC algorithm."""
        return self._run_python(X_views)

    def _run_python(self, X_views):
        """
        Python implementation of ALPC algorithm.
        
        Based on the MATLAB code from the paper with improved initialization.
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
        from scipy.spatial.distance import cdist
        
        n_views = len(X_views)
        n_samples = X_views[0].shape[0]
        k = self.n_clusters
        m = self.n_anchors_per_cluster
        n_anchors = m * k  # Total number of anchors
        
        alpha = self.alpha
        beta = self.beta
        reg = 1e-4  # Regularization for numerical stability
        
        # Transpose X_views to match MATLAB convention (features x samples)
        X_T = [X.T.astype(np.float64) for X in X_views]  # Each is (d_v, n_samples)
        
        # Better initialization: use K-means to initialize anchors
        # Concatenate views for initial anchor selection
        X_concat = np.hstack(X_views)  # (n_samples, sum_dims)
        
        # Run K-means to get initial anchor positions
        kmeans_init = KMeans(n_clusters=n_anchors, random_state=self.random_state, n_init=3, max_iter=30)
        kmeans_init.fit(X_concat)
        
        # Initialize Z based on soft assignment to K-means centers
        centers = kmeans_init.cluster_centers_  # (n_anchors, sum_dims)
        D = cdist(X_concat, centers, 'euclidean')
        sigma = np.median(D) + 1e-10
        S = np.exp(-D**2 / (2 * sigma**2))
        Z = (S / (S.sum(axis=1, keepdims=True) + 1e-10)).T  # (n_anchors, n_samples)
        
        # Initialize anchor matrices A_v from projected K-means centers
        A_views = []
        dim_offset = 0
        for v in range(n_views):
            d_v = X_views[v].shape[1]
            # Extract view-specific part of centers
            A_v = centers[:, dim_offset:dim_offset + d_v].T  # (d_v, n_anchors)
            A_views.append(A_v.astype(np.float64))
            dim_offset += d_v
        
        # Initialize R (k x n_samples) - cluster indicator via K-means on Z
        Z_for_cluster = Z.T  # (n_samples, n_anchors)
        kmeans_r = KMeans(n_clusters=k, random_state=self.random_state, n_init=3)
        labels_init = kmeans_r.fit_predict(Z_for_cluster)
        R = np.zeros((k, n_samples))
        for i, label in enumerate(labels_init):
            R[label, i] = 1.0
        
        # Initialize Y (k x n_anchors) - anchor-to-cluster assignment
        # Each cluster has m anchors
        Y = np.zeros((k, n_anchors))
        for ik in range(k):
            Y[ik, ik*m:(ik+1)*m] = 1
        
        # Initialize P = Y
        P = Y.copy().astype(np.float64)
        
        # Initialize U_v via SVD of A_v @ P^T
        U_views = []
        for v in range(n_views):
            AP = A_views[v] @ P.T
            if np.linalg.norm(AP) > 1e-10:
                Up, _, Vt = np.linalg.svd(AP, full_matrices=False)
                U_views.append(Up @ Vt)
            else:
                U_views.append(np.eye(min(A_views[v].shape[0], k)))
        
        obj_prev = np.inf
        for iteration in range(self.max_iter):
            # Update Z (n_anchors x n_samples)
            tZ1 = reg * np.eye(n_anchors)
            tZ2 = np.zeros((n_anchors, n_samples))
            for v in range(n_views):
                tZ1 += A_views[v].T @ A_views[v]
                tZ2 += A_views[v].T @ X_T[v]
            
            try:
                Z = np.linalg.solve(tZ1 + beta * np.eye(n_anchors), tZ2 + beta * P.T @ R)
            except np.linalg.LinAlgError:
                Z = np.linalg.lstsq(tZ1 + beta * np.eye(n_anchors), tZ2 + beta * P.T @ R, rcond=None)[0]
            
            # Update A_v for each view
            ZZT = Z @ Z.T + (alpha + reg) * np.eye(n_anchors)
            ZZT_inv = np.linalg.inv(ZZT)
            for v in range(n_views):
                A_views[v] = (X_T[v] @ Z.T + alpha * U_views[v] @ P) @ ZZT_inv
            
            # Update U_v for each view (SVD)
            for v in range(n_views):
                AP = A_views[v] @ P.T
                if np.linalg.norm(AP) > 1e-10:
                    Up, _, Vt = np.linalg.svd(AP, full_matrices=False)
                    U_views[v] = Up @ Vt
            
            # Update P
            UA = np.zeros((k, n_anchors))
            for v in range(n_views):
                if U_views[v].shape[0] >= k:
                    UA += U_views[v][:k, :k].T @ A_views[v][:k, :] if A_views[v].shape[0] >= k else U_views[v].T @ A_views[v]
                else:
                    UA += U_views[v].T @ A_views[v]
            
            RRT = R @ R.T + reg * np.eye(k)
            try:
                P = np.linalg.solve(alpha * np.eye(k) + beta * RRT, UA + beta * R @ Z.T)
            except np.linalg.LinAlgError:
                P = np.linalg.lstsq(alpha * np.eye(k) + beta * RRT, UA + beta * R @ Z.T, rcond=None)[0]
            
            # Update R
            PPT = P @ P.T + reg * np.eye(k)
            try:
                PPT_inv = np.linalg.inv(PPT)
                R = PPT_inv @ P @ Z
            except np.linalg.LinAlgError:
                R = np.linalg.lstsq(PPT, P @ Z, rcond=None)[0]
            
            # Compute objective (for convergence check)
            obj = 0
            for v in range(n_views):
                obj += np.linalg.norm(X_T[v] - A_views[v] @ Z, 'fro') ** 2
                obj += alpha * np.linalg.norm(A_views[v] - U_views[v] @ P, 'fro') ** 2
            obj += beta * np.linalg.norm(Z - P.T @ R, 'fro') ** 2
            
            # Check convergence
            if iteration > 0 and abs(obj_prev - obj) / (abs(obj_prev) + 1e-10) < 1e-6:
                break
            obj_prev = obj
        
        # Get cluster assignments from R (cluster indicator matrix)
        # R is (k x n_samples), each column indicates cluster membership
        labels = np.argmax(R, axis=0)
        
        # If R doesn't give good assignments, use K-means on Z
        if len(np.unique(labels)) < k:
            # Z is (n_anchors x n_samples), use Z^T for clustering
            Z_embedding = Z.T  # (n_samples x n_anchors)
            Z_embedding = normalize(Z_embedding, axis=1)
            
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(Z_embedding)
        
        return labels

    def _run_fallback(self, X_views):
        """Fallback to Python implementation."""
        return self._run_python(X_views)


class RCAGLWrapper(BaseExternalMVC):
    """
    RCAGL: Robust and Consistent Anchor Graph Learning for Multi-View Clustering.
    
    Paper: "Robust and Consistent Anchor Graph Learning for Multi-View Clustering"
    Venue: IEEE Transactions on Knowledge and Data Engineering (TKDE) 2024
    GitHub: https://github.com/Tracesource/RCAGL
    
    Setup:
        cd spock/baselines/external
        git clone https://github.com/Tracesource/RCAGL.git
    
    Requirements:
        - scikit-learn
        - scipy
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    n_anchors : int, default=None
        Number of anchors. If None, uses 2*n_clusters.
    lambda_param : float, default=100
        Regularization parameter.
    max_iter : int, default=50
        Maximum number of iterations.
    random_state : int, optional
        Random seed.
    """
    
    def __init__(self, n_clusters, n_anchors=None, lambda_param=100, 
                 max_iter=50, random_state=None):
        super().__init__(n_clusters, device='cpu', random_state=random_state)
        self.n_anchors = n_anchors if n_anchors else 2 * n_clusters
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.name = 'RCAGL'
    
    def _get_external_paths(self):
        return ['RCAGL', 'rcagl']
    
    def _run_external(self, X_views):
        """Run RCAGL algorithm (Python implementation)."""
        return self._run_python(X_views)
    
    def _run_python(self, X_views):
        """
        Python implementation of RCAGL algorithm.
        
        RCAGL jointly learns:
        - A: View-specific projection matrices
        - C: Shared anchor graph
        - D: View-specific refinement matrices
        
        Optimization:
        min_A,C,D sum_v ||X_v - 0.5*A_v*(C+D_v)||_F^2 + lambda*sum_v ||D_v||_F^2
        s.t. A_v'*A_v = I, C and D_v are non-negative with row-sum = 1
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from scipy.linalg import svd
        from scipy.sparse.linalg import svds
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_views = len(X_views)
        n_samples = X_views[0].shape[0]
        m = self.n_anchors  # number of anchors
        k = self.n_clusters
        lam = self.lambda_param
        
        # Normalize each view
        X_normalized = []
        for v in range(n_views):
            scaler = StandardScaler()
            X_v = scaler.fit_transform(X_views[v])
            X_normalized.append(X_v.T)  # (d_v x n)
        
        # Initialize C via K-means on concatenated SVD features
        X_concat = np.vstack(X_normalized)  # (sum_d x n)
        
        # SVD for dimensionality reduction
        n_svd = min(m, X_concat.shape[0], X_concat.shape[1] - 1)
        U, s, Vt = svds(X_concat, k=n_svd)
        X_reduced = Vt.T  # (n x m)
        
        # K-means to get initial cluster assignments
        kmeans = KMeans(n_clusters=m, n_init=10, max_iter=100, random_state=self.random_state)
        idx = kmeans.fit_predict(X_reduced)
        
        # Initialize C as indicator matrix (m x n)
        C = np.zeros((m, n_samples))
        for i in range(n_samples):
            C[idx[i], i] = 1.0
        
        # Initialize A and D for each view
        A = []
        D = []
        dims = []
        for v in range(n_views):
            d_v = X_normalized[v].shape[0]
            dims.append(d_v)
            A.append(np.zeros((d_v, m)))
            D.append(C.copy())  # Initialize D same as C
        
        # Simplex projection (EProjSimplex)
        def proj_simplex(v):
            """Project vector onto probability simplex."""
            n = len(v)
            u = np.sort(v)[::-1]
            cssv = np.cumsum(u) - 1
            ind = np.arange(1, n + 1)
            cond = u - cssv / ind > 0
            if np.any(cond):
                rho = ind[cond][-1]
                theta = cssv[cond][-1] / rho
            else:
                rho = 1
                theta = (np.sum(v) - 1) / n
            return np.maximum(v - theta, 0)
        
        obj_prev = np.inf
        
        for iteration in range(self.max_iter):
            # Update A_v (orthogonal projection)
            for v in range(n_views):
                G = X_normalized[v] @ (C + D[v]).T  # (d_v x m)
                U_a, _, Vh_a = svd(G, full_matrices=False)
                A[v] = U_a @ Vh_a  # Orthogonal Procrustes
            
            # Update D_v
            for v in range(n_views):
                H = 2 * (X_normalized[v] - 0.5 * A[v] @ C).T @ A[v]  # (n x m)
                D_v_new = np.zeros((m, n_samples))
                for i in range(n_samples):
                    ut = H[i, :] / (1 + 4 * lam)
                    D_v_new[:, i] = proj_simplex(ut)
                D[v] = D_v_new
            
            # Update C via bipartite co-clustering
            B = np.zeros((n_samples, m))
            G_avg = np.zeros((m, n_samples))
            for v in range(n_views):
                B += (X_normalized[v] - 0.5 * A[v] @ D[v]).T @ A[v]
                G_avg += D[v]
            B /= n_views
            G_avg /= n_views
            
            # Simplified spectral clustering on bipartite graph
            # Construct normalized affinity
            S = B  # (n x m)
            d1 = np.abs(S).sum(axis=1) + 1e-10
            d2 = np.abs(S).sum(axis=0) + 1e-10
            D1_inv_sqrt = np.diag(1.0 / np.sqrt(d1))
            D2_inv_sqrt = np.diag(1.0 / np.sqrt(d2))
            
            S_normalized = D1_inv_sqrt @ S @ D2_inv_sqrt
            
            # SVD for spectral embedding
            try:
                U_s, s_s, Vt_s = svds(S_normalized, k=min(k, m-1, n_samples-1))
                # Sort by singular values (descending)
                sort_idx = np.argsort(s_s)[::-1]
                U_s = U_s[:, sort_idx]
            except:
                U_s = np.random.randn(n_samples, k)
            
            # Normalize rows
            U_embed = U_s[:, :k]
            row_norms = np.linalg.norm(U_embed, axis=1, keepdims=True) + 1e-10
            U_embed = U_embed / row_norms
            
            # K-means on spectral embedding
            kmeans_c = KMeans(n_clusters=m, n_init=10, random_state=self.random_state)
            try:
                idx_new = kmeans_c.fit_predict(U_embed)
            except:
                idx_new = idx
            
            # Update C
            C_new = np.zeros((m, n_samples))
            for i in range(n_samples):
                C_new[idx_new[i] % m, i] = 1.0
            C = C_new
            
            # Compute objective
            obj = 0
            for v in range(n_views):
                obj += np.linalg.norm(X_normalized[v] - 0.5 * A[v] @ (C + D[v]), 'fro') ** 2
                obj += lam * np.linalg.norm(D[v], 'fro') ** 2
            
            # Check convergence (handle inf/nan cases)
            if np.isfinite(obj) and np.isfinite(obj_prev):
                rel_change = abs(obj_prev - obj) / (abs(obj_prev) + 1e-10)
                if rel_change < 1e-4:
                    break
            obj_prev = obj
        
        # Final clustering from C
        # Aggregate information for final assignment
        # Use D matrices average for soft assignment
        D_avg = np.zeros((m, n_samples))
        for v in range(n_views):
            D_avg += D[v]
        D_avg /= n_views
        
        # Spectral clustering on the combined graph
        S_final = (C + D_avg).T  # (n x m)
        
        # Normalized spectral clustering
        d1 = S_final.sum(axis=1) + 1e-10
        D1_sqrt_inv = np.diag(1.0 / np.sqrt(d1))
        L = D1_sqrt_inv @ S_final
        
        try:
            U_final, _, _ = svds(L, k=min(k, m-1))
            U_final = U_final[:, ::-1]  # Reverse to get largest first
        except:
            U_final = np.random.randn(n_samples, k)
        
        # Normalize
        row_norms = np.linalg.norm(U_final, axis=1, keepdims=True) + 1e-10
        U_final = U_final / row_norms
        
        # K-means clustering
        kmeans_final = KMeans(n_clusters=k, n_init=100, random_state=self.random_state)
        labels = kmeans_final.fit_predict(U_final)
        
        return labels
    
    def _run_fallback(self, X_views):
        """Fallback to Python implementation."""
        return self._run_python(X_views)


class ROLLWrapper(BaseExternalMVC):
    """
    ROLL: Robust Noisy Pseudo-label Learning for Multi-View Clustering with Noisy Correspondence.
    
    Paper: "ROLL: Robust Noisy Pseudo-label Learning for Multi-View Clustering 
           with Noisy Correspondence" (CVPR 2025)
    GitHub: https://github.com/sunyuan-cs/2025-CVPR-ROLL
    
    Setup:
        cd spock/baselines/external
        git clone https://github.com/sunyuan-cs/2025-CVPR-ROLL.git
    
    Requirements:
        - PyTorch >= 1.5.0
        - scikit-learn
        - munkres
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    warm_epochs : int, default=50
        Number of warmup epochs (contrastive pretraining).
    epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=128
        Batch size for training.
    learning_rate : float, default=1e-4
        Learning rate.
    tau : float, default=0.5
        Temperature for contrastive loss.
    alpha : float, default=1e-4
        Weight for robust contrastive loss.
    beta : float, default=1e-2
        Weight for pseudo-label loss.
    q : float, default=0.1
        q parameter for GCE loss.
    feature_dim : int, default=512
        Hidden feature dimension.
    device : str, default='auto'
        Device to use ('cuda', 'cpu', or 'auto').
    random_state : int, optional
        Random seed.
    """
    
    def __init__(self, n_clusters, warm_epochs=50, epochs=100, batch_size=128,
                 learning_rate=1e-4, tau=0.5, alpha=1e-4, beta=1e-2, q=0.1,
                 feature_dim=512, device='auto', random_state=None):
        super().__init__(n_clusters, device, random_state)
        self.warm_epochs = warm_epochs
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.feature_dim = feature_dim
        self.name = 'ROLL'
    
    def _get_external_paths(self):
        return ['2025-CVPR-ROLL', 'ROLL', 'roll']
    
    def _run_external(self, X_views):
        """Run ROLL algorithm with external code components."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.cluster import KMeans
        
        device = torch.device(self.device)
        n_views = len(X_views)
        n_samples = X_views[0].shape[0]
        dims = [x.shape[1] for x in X_views]
        k = self.n_clusters
        
        # Normalize data
        X_normalized = []
        for x in X_views:
            scaler = MinMaxScaler()
            X_normalized.append(scaler.fit_transform(x).astype(np.float32))
        
        # Build flexible encoder/decoder model for arbitrary input dimensions
        class FlexibleROLLModel(nn.Module):
            def __init__(self, input_dims, feature_dim=512):
                super().__init__()
                self.n_views = len(input_dims)
                self.feature_dim = feature_dim
                
                # Create encoders for each view
                self.encoders = nn.ModuleList()
                for dim in input_dims:
                    encoder = nn.Sequential(
                        nn.Linear(dim, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(True),
                        nn.Dropout(0.1),
                        nn.Linear(1024, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(True),
                        nn.Dropout(0.1),
                        nn.Linear(1024, feature_dim),
                        nn.BatchNorm1d(feature_dim),
                        nn.ReLU(True)
                    )
                    self.encoders.append(encoder)
                
                # Create decoders for each view (from concatenated features)
                self.decoders = nn.ModuleList()
                concat_dim = self.n_views * feature_dim
                for dim in input_dims:
                    decoder = nn.Sequential(
                        nn.Linear(concat_dim, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(1024, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(1024, dim)
                    )
                    self.decoders.append(decoder)
            
            def forward(self, *views):
                # Encode each view
                h_list = []
                for i, x in enumerate(views):
                    h = self.encoders[i](x)
                    h = F.normalize(h, dim=1)
                    h_list.append(h)
                
                # Concatenate for reconstruction
                union = torch.cat(h_list, dim=1)
                
                # Decode each view
                z_list = []
                for i in range(self.n_views):
                    z = self.decoders[i](union)
                    z_list.append(z)
                
                return h_list, z_list
        
        # Create dataset
        class MVDataset(Dataset):
            def __init__(self, views):
                self.views = [torch.FloatTensor(v) for v in views]
                self.n_samples = views[0].shape[0]
            
            def __len__(self):
                return self.n_samples
            
            def __getitem__(self, idx):
                return tuple(v[idx] for v in self.views), idx
        
        dataset = MVDataset(X_normalized)
        data_loader = DataLoader(dataset, batch_size=min(self.batch_size, n_samples),
                                  shuffle=True, drop_last=True)
        full_loader = DataLoader(dataset, batch_size=n_samples, shuffle=False)
        
        # Initialize model
        model = FlexibleROLLModel(dims, self.feature_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        mse_loss = nn.MSELoss()
        
        # Phase 1: Warmup (contrastive pretraining)
        model.train()
        for epoch in range(self.warm_epochs):
            for batch_data, idx in data_loader:
                views_batch = [v.to(device) for v in batch_data]
                
                optimizer.zero_grad()
                h_list, z_list = model(*views_batch)
                
                # Reconstruction loss
                recon_loss = sum(mse_loss(views_batch[i], z_list[i]) for i in range(n_views))
                
                # Contrastive loss between views
                if n_views >= 2:
                    h0, h1 = h_list[0], h_list[1]
                    I = torch.eye(h0.size(0)).to(device)
                    sim = h0.mm(h1.t()) / self.tau
                    cl_loss = -(I * sim.softmax(1).log()).sum(1).mean()
                else:
                    cl_loss = torch.tensor(0.0).to(device)
                
                loss = recon_loss + 1e-4 * cl_loss
                loss.backward()
                optimizer.step()
        
        # Get initial embeddings for pseudo-label generation
        model.eval()
        with torch.no_grad():
            for batch_data, _ in full_loader:
                views_full = [v.to(device) for v in batch_data]
                h_list_full, _ = model(*views_full)
        
        # Generate pseudo-labels using K-means on concatenated features
        h_concat = torch.cat(h_list_full, dim=1).cpu().numpy()
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
        pseudo_labels = kmeans.fit_predict(h_concat)
        
        # Create view-specific class centers by computing mean of each view's features per cluster
        class_centers_list = []
        for v in range(n_views):
            h_v = h_list_full[v].cpu().numpy()
            centers_v = np.zeros((k, self.feature_dim))
            for c in range(k):
                mask = pseudo_labels == c
                if mask.sum() > 0:
                    centers_v[c] = h_v[mask].mean(axis=0)
            class_centers_list.append(F.normalize(torch.FloatTensor(centers_v), dim=1).to(device))
        
        # Create soft pseudo-labels
        def create_soft_labels(h, centers, temperature=0.1):
            sim = h.mm(centers.t()) / temperature
            return F.softmax(sim, dim=1)
        
        # Phase 2: Main training with pseudo-labels
        model.train()
        for epoch in range(self.epochs):
            for batch_data, idx in data_loader:
                views_batch = [v.to(device) for v in batch_data]
                batch_pseudo = torch.LongTensor(pseudo_labels[idx.numpy()]).to(device)
                
                optimizer.zero_grad()
                h_list, z_list = model(*views_batch)
                
                # Reconstruction loss
                recon_loss = sum(mse_loss(views_batch[i], z_list[i]) for i in range(n_views))
                
                # Pseudo-label loss for each view
                pl_loss = torch.tensor(0.0).to(device)
                for v_idx, h in enumerate(h_list):
                    soft_pred = create_soft_labels(h, class_centers_list[v_idx])
                    soft_target = F.one_hot(batch_pseudo, k).float()
                    
                    # Compute adaptive weights based on similarity
                    sim_diag = (soft_pred * soft_target).sum(1)
                    w = sim_diag / (sim_diag.sum() + 1e-10)
                    
                    # Weighted cross-entropy
                    pl_loss = pl_loss - self.beta * (w * (soft_target * soft_pred.log()).sum(1)).mean()
                
                # Robust contrastive loss (GCE-style)
                if n_views >= 2:
                    h0, h1 = h_list[0], h_list[1]
                    cos_sim = h0.mm(h1.t())
                    sim = (cos_sim / self.tau).exp()
                    pos = sim.diag()
                    q = self.q
                    rcl_loss = (((1 - q) * (sim.sum(1))**q - pos**q) / q).mean()
                    rcl_loss = self.alpha * rcl_loss
                else:
                    rcl_loss = torch.tensor(0.0).to(device)
                
                loss = recon_loss + pl_loss + rcl_loss
                loss.backward()
                optimizer.step()
            
            # Update pseudo-labels periodically
            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    for batch_data, _ in full_loader:
                        views_full = [v.to(device) for v in batch_data]
                        h_list_full, _ = model(*views_full)
                
                h_concat = torch.cat(h_list_full, dim=1).cpu().numpy()
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
                pseudo_labels = kmeans.fit_predict(h_concat)
                # Update view-specific class centers
                for v in range(n_views):
                    h_v = h_list_full[v].cpu().numpy()
                    centers_v = np.zeros((k, self.feature_dim))
                    for c in range(k):
                        mask = pseudo_labels == c
                        if mask.sum() > 0:
                            centers_v[c] = h_v[mask].mean(axis=0)
                    class_centers_list[v] = F.normalize(torch.FloatTensor(centers_v), dim=1).to(device)
                model.train()
        
        # Final clustering
        model.eval()
        with torch.no_grad():
            for batch_data, _ in full_loader:
                views_full = [v.to(device) for v in batch_data]
                h_list_full, _ = model(*views_full)
        
        h_concat = torch.cat(h_list_full, dim=1).cpu().numpy()
        kmeans = KMeans(n_clusters=k, n_init=100, random_state=self.random_state)
        labels = kmeans.fit_predict(h_concat)
        
        return labels


class DMACWrapper(BaseExternalMVC):
    """
    DMAC: Deep Multi-view Anchor Clustering.
    
    Paper: "Towards Learnable Anchor for Deep Multi-View Clustering" (AAAI 2025)
    GitHub: https://github.com/drongwbc/DMAC-AAAI25
    
    Setup:
        cd spock/baselines/external
        git clone https://github.com/drongwbc/DMAC-AAAI25.git
    
    Requirements:
        - PyTorch >= 2.0.0
        - scikit-learn
        - scipy
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    hidden_size : int, default=256
        Hidden layer size in encoder/decoder.
    emblem_size : int, default=64
        Embedding dimension.
    pre_epochs : int, default=100
        Number of pretraining epochs.
    epochs : int, default=100
        Number of training epochs.
    lr1 : float, default=1e-3
        Learning rate for pretraining.
    lr2 : float, default=1e-4
        Learning rate for training.
    alpha : float, default=0.1
        Weight for structure loss.
    beta : float, default=0.1
        Weight for mutual information loss.
    device : str, default='auto'
        Device to use ('cuda', 'cpu', or 'auto').
    random_state : int, optional
        Random seed.
    """
    
    def __init__(self, n_clusters, hidden_size=256, emblem_size=64,
                 pre_epochs=100, epochs=100, lr1=1e-3, lr2=1e-4,
                 alpha=0.1, beta=0.1, device='auto', random_state=None):
        super().__init__(n_clusters, device, random_state)
        self.hidden_size = hidden_size
        self.emblem_size = emblem_size
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.lr1 = lr1
        self.lr2 = lr2
        self.alpha = alpha
        self.beta = beta
        self.name = 'DMAC'
    
    def _get_external_paths(self):
        return ['DMAC-AAAI25', 'DMAC', 'dmac']
    
    def _run_external(self, X_views):
        """Run DMAC algorithm."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.optim import RMSprop
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
        import math
        
        device = torch.device(self.device)
        n_views = len(X_views)
        n_samples = X_views[0].shape[0]
        dims = [x.shape[1] for x in X_views]
        k = self.n_clusters
        
        # Normalize data
        X_normalized = []
        for x in X_views:
            x_norm = normalize(x, norm='l2')
            X_normalized.append(torch.FloatTensor(x_norm).to(device))
        
        # Compute anchor number
        anchor_num = math.floor(math.sqrt(n_samples * k))
        neighbor = 5
        init_nei = neighbor
        upper = math.floor(anchor_num / k)
        
        # Helper functions from DMAC
        def distance(X, Y, square=True):
            n = X.shape[1]
            m = Y.shape[1]
            x = torch.norm(X, dim=0)
            x = x * x
            x = torch.t(x.repeat(m, 1))
            y = torch.norm(Y, dim=0)
            y = y * y
            y = y.repeat(n, 1)
            crossing_term = torch.t(X).matmul(Y)
            result = x + y - 2 * crossing_term
            result = result.relu()
            if not square:
                result = torch.sqrt(result)
            return result
        
        def construct_anchor_graph(Z, anchor, k_neighbor):
            distances = distance(Z.t(), anchor.t(), square=True)
            sorted_distances, _ = distances.sort(dim=1)
            top_k = sorted_distances[:, k_neighbor]
            top_k = torch.t(top_k.repeat(distances.shape[1], 1)) + 1e-10
            sum_top_k = torch.sum(sorted_distances[:, 0:k_neighbor], dim=1)
            sum_top_k = torch.t(sum_top_k.repeat(distances.shape[1], 1))
            weights = torch.div(top_k - distances, k_neighbor * top_k - sum_top_k + 1e-7)
            weights = weights.relu()
            return weights
        
        def anchor_selection_loss(Z, anchor, v=1):
            q = 1.0 / (1.0 + torch.sum(
                torch.pow(Z.unsqueeze(1) - anchor, 2), 2) / v)
            q = q.pow((v + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
            q = torch.clamp(q, min=1e-12)
            loss = torch.mean(-torch.sum(q * torch.log(q), dim=1))
            return loss
        
        def mi_loss(Q1, Q2):
            joint_prob = Q1.unsqueeze(2) * Q2.unsqueeze(1)
            p_marginal = joint_prob.sum(dim=2, keepdim=True)
            q_marginal = joint_prob.sum(dim=1, keepdim=True)
            joint_prob = torch.clamp(joint_prob, min=1e-10)
            p_marginal = torch.clamp(p_marginal, min=1e-10)
            q_marginal = torch.clamp(q_marginal, min=1e-10)
            mi = (joint_prob * torch.log(joint_prob / (p_marginal * q_marginal))).sum(dim=(1, 2))
            return -mi.mean()
        
        def structure_loss(Z, A):
            p_D = torch.pinverse(torch.diag(torch.sum(A, dim=0)))
            t = torch.mm(Z.T, Z) - torch.mm(torch.mm(torch.mm(torch.mm(Z.T, A), p_D), A.T), Z)
            loss = torch.trace(t)
            return loss.mean()
        
        # DMAC Model
        class DMACModel(nn.Module):
            def __init__(self, num_features, hidden_size, emblem_size, views, n_clusters):
                super().__init__()
                self.views = views
                
                # Encoder
                self.encoder_l1 = nn.ModuleList([
                    nn.Linear(num_features[i], hidden_size, bias=True) for i in range(views)
                ])
                self.encoder_l2 = nn.ModuleList([
                    nn.Linear(hidden_size, emblem_size, bias=True) for i in range(views)
                ])
                
                # Decoder
                self.decoder_l1 = nn.ModuleList([
                    nn.Linear(emblem_size, hidden_size, bias=True) for i in range(views)
                ])
                self.decoder_l2 = nn.ModuleList([
                    nn.Linear(hidden_size, num_features[i], bias=True) for i in range(views)
                ])
                
                # Noise Generator
                self.mu = nn.Linear(emblem_size, emblem_size, bias=True)
                self.zita = nn.Linear(emblem_size, emblem_size, bias=True)
                self.prelu_weight = nn.Parameter(torch.Tensor(emblem_size).fill_(0.25))
                
                # Anchor Graph Convolution
                self.agcn_l1 = nn.ModuleList([
                    nn.Linear(emblem_size, int(emblem_size / 4), bias=False) for i in range(views)
                ])
                self.agcn_l2 = nn.ModuleList([
                    nn.Linear(int(emblem_size / 4), n_clusters, bias=False) for i in range(views)
                ])
            
            def anchor_convolution(self, A, F_mat):
                p_D = torch.pinverse(torch.diag(torch.sum(A, dim=0)))
                t1 = torch.mm(A, F_mat)
                t2 = torch.mm(A.T, t1)
                t3 = torch.mm(p_D, t2)
                return t3
            
            def forward(self, X, anchor_num, neighbor_k, preTrain=False):
                # Encoder
                Z = []
                for i in range(self.views):
                    tmp_z = F.leaky_relu(self.encoder_l1[i](X[i]))
                    tmp_z = F.leaky_relu(self.encoder_l2[i](tmp_z))
                    tmp_z = F.normalize(tmp_z, p=2, dim=1)
                    Z.append(tmp_z)
                
                # Fusion
                fuse_Z = sum(Z) / self.views
                fuse_Z = F.normalize(fuse_Z, p=2, dim=1)
                
                if preTrain:
                    rec_X = []
                    for i in range(self.views):
                        tmp_X = F.leaky_relu(self.decoder_l1[i](fuse_Z))
                        tmp_X = F.leaky_relu(self.decoder_l2[i](tmp_X))
                        tmp_X = F.normalize(tmp_X, p=2, dim=1)
                        rec_X.append(tmp_X)
                    return rec_X
                
                # Learnable anchor with noise
                anchor = KMeans(n_clusters=anchor_num, n_init=1,
                               init='k-means++').fit(fuse_Z.detach().cpu().numpy()).cluster_centers_
                anchor = torch.tensor(anchor, requires_grad=True, dtype=torch.float32).to(fuse_Z.device)
                
                mvn = torch.distributions.MultivariateNormal(
                    torch.zeros(anchor.size(1)), torch.eye(anchor.size(1)))
                noise = mvn.sample((anchor.size(0),)).to(anchor.device)
                mu = F.prelu(self.mu(anchor), self.prelu_weight)
                zita = F.relu(self.zita(anchor))
                anchor = anchor + (mu + torch.mul(zita, noise))
                anchor = F.normalize(anchor, p=2, dim=1)
                
                # Anchor graph learning
                A = []
                for i in range(self.views):
                    tmp_A = construct_anchor_graph(Z[i], anchor, neighbor_k)
                    A.append(tmp_A)
                
                # Anchor graph convolution
                Q = []
                for i in range(self.views):
                    tmp_Q = F.leaky_relu(self.agcn_l1[i](self.anchor_convolution(A[i], anchor)))
                    tmp_Q = F.softmax(self.agcn_l2[i](self.anchor_convolution(A[i], tmp_Q)), dim=1)
                    Q.append(tmp_Q)
                
                return Z, anchor, fuse_Z, A, Q
        
        # Initialize model
        model = DMACModel(dims, self.hidden_size, self.emblem_size, n_views, k).to(device)
        
        # Pretraining
        if self.pre_epochs > 0:
            optimizer = RMSprop(model.parameters(), lr=self.lr1)
            mse = nn.MSELoss()
            model.train()
            for epoch in range(self.pre_epochs):
                rec_X = model(X_normalized, anchor_num, neighbor, preTrain=True)
                loss = sum(mse(X_normalized[i], rec_X[i]) for i in range(n_views))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Main training
        optimizer = RMSprop(model.parameters(), lr=self.lr2)
        model.train()
        for epoch in range(self.epochs):
            Z, anchor, fuse_Z, A, Q = model(X_normalized, anchor_num, neighbor)
            
            l1 = sum(anchor_selection_loss(Z[i], anchor) for i in range(n_views))
            l2 = sum(structure_loss(fuse_Z, A[i]) for i in range(n_views))
            l3 = torch.tensor(0.0).to(device)
            for i in range(n_views):
                for j in range(i + 1, n_views):
                    l3 = l3 + mi_loss(Q[i], Q[j])
            
            loss = l1 + self.alpha * l2 + self.beta * l3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update neighbor periodically
            if epoch > 0 and epoch % 20 == 0:
                neighbor = min(neighbor + init_nei, upper)
        
        # Final clustering
        model.eval()
        with torch.no_grad():
            Z, anchor, fuse_Z, A, Q = model(X_normalized, anchor_num, neighbor)
        
        labels = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit_predict(
            fuse_Z.cpu().numpy()
        )
        
        return labels


# Registry of external methods
EXTERNAL_METHODS = {
    'SCMVC': SCMVCWrapper,
    'EFIMVC': EFIMVCWrapper,
    'ALPC': ALPCWrapper,
    'RCAGL': RCAGLWrapper,
    'ROLL': ROLLWrapper,
    'DMAC': DMACWrapper,
}


def get_scalable_methods(n_clusters, n_anchors=500, random_state=None, methods=None,
                         include_external=False):
    """
    Get scalable multi-view clustering methods.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    n_anchors : int, default=500
        Number of anchors for anchor-based methods.
    random_state : int, optional
        Random seed.
    methods : list, optional
        List of method names to include. If None, includes all built-in methods.
    include_external : bool, default=False
        Whether to include external methods (require git clone).

    Returns
    -------
    methods_dict : dict
        Dictionary of method name -> method instance.
    """
    if methods is None:
        methods = list(SCALABLE_METHODS.keys())
        if include_external:
            methods.extend(EXTERNAL_METHODS.keys())

    result = {}
    for name in methods:
        # Check built-in methods first
        if name in SCALABLE_METHODS:
            cls = SCALABLE_METHODS[name]
            if name in ['LMVSC', 'SMVSC', 'EOMSC-CA']:
                result[name] = cls(n_clusters, n_anchors=n_anchors, random_state=random_state)
            else:
                result[name] = cls(n_clusters, random_state=random_state)
        # Check external methods
        elif name in EXTERNAL_METHODS:
            cls = EXTERNAL_METHODS[name]
            result[name] = cls(n_clusters, random_state=random_state)

    return result


def check_scalable_methods_availability():
    """
    Check which scalable methods are available.

    Returns
    -------
    availability : dict
        Dictionary of method name -> bool.
        Built-in methods are always True.
        External methods are True if their code is found.
    """
    result = {name: True for name in SCALABLE_METHODS}

    # Check external methods
    for name, cls in EXTERNAL_METHODS.items():
        wrapper = cls(n_clusters=2)  # Dummy instance to check paths
        result[name] = wrapper._find_external_path() is not None

    return result


def list_external_methods():
    """
    List external methods and their setup instructions.

    Returns
    -------
    info : dict
        Dictionary of method name -> setup info.
    """
    return {
        'SCMVC': {
            'name': 'Self-Weighted Contrastive Fusion for Deep MVC',
            'paper': 'IEEE TMM 2024',
            'github': 'https://github.com/SongwuJob/SCMVC',
            'setup': f'cd {EXTERNAL_DIR} && git clone https://github.com/SongwuJob/SCMVC.git',
            'requirements': ['torch>=1.12.0', 'scikit-learn'],
        },
        'EFIMVC': {
            'name': 'Efficient Federated Incomplete Multi-View Clustering',
            'paper': 'ICML 2025',
            'github': 'https://github.com/Tracesource/EFIMVC',
            'setup': f'cd {EXTERNAL_DIR} && git clone https://github.com/Tracesource/EFIMVC.git',
            'requirements': ['scikit-learn', 'scipy'],
            'note': 'Original MATLAB implementation. Python approximation used as fallback.',
        },
        'ALPC': {
            'name': 'Anchor Learning with Potential Cluster Constraints for MVC',
            'paper': 'AAAI 2025',
            'github': 'https://github.com/whbdmu/ALPC',
            'setup': f'cd {EXTERNAL_DIR} && git clone https://github.com/whbdmu/ALPC.git',
            'requirements': ['scikit-learn', 'scipy'],
            'note': 'Original MATLAB implementation. Python implementation provided.',
        },
        'RCAGL': {
            'name': 'Robust and Consistent Anchor Graph Learning for MVC',
            'paper': 'IEEE TKDE 2024',
            'github': 'https://github.com/Tracesource/RCAGL',
            'setup': f'cd {EXTERNAL_DIR} && git clone https://github.com/Tracesource/RCAGL.git',
            'requirements': ['scikit-learn', 'scipy'],
            'note': 'Original MATLAB implementation. Python implementation provided.',
        },
        'ROLL': {
            'name': 'Robust Noisy Pseudo-label Learning for MVC with Noisy Correspondence',
            'paper': 'CVPR 2025',
            'github': 'https://github.com/sunyuan-cs/2025-CVPR-ROLL',
            'setup': f'cd {EXTERNAL_DIR} && git clone https://github.com/sunyuan-cs/2025-CVPR-ROLL.git',
            'requirements': ['torch>=1.5.0', 'scikit-learn', 'munkres'],
            'note': 'Handles noisy correspondences. Python implementation provided.',
        },
    }


# For backwards compatibility
DEEP_METHODS = {**SCALABLE_METHODS, **EXTERNAL_METHODS}
get_deep_methods = get_scalable_methods
check_deep_methods_availability = check_scalable_methods_availability
