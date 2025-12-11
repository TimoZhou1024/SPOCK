"""
Baseline Multi-View Clustering Methods for Comparison

Implements several SOTA methods for fair comparison with SPOCK.
"""

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import eigh
import warnings


class BaseMultiViewClustering:
    """Base class for multi-view clustering methods."""
    
    def __init__(self, n_clusters, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
    
    def fit(self, X_views):
        raise NotImplementedError
    
    def fit_predict(self, X_views):
        self.fit(X_views)
        return self.labels_


class ConcatKMeans(BaseMultiViewClustering):
    """
    Concatenation + K-Means baseline.
    
    Simple baseline that concatenates all views and runs K-Means.
    """
    
    def __init__(self, n_clusters, random_state=None):
        super().__init__(n_clusters, random_state)
        self.name = 'Concat+KMeans'
    
    def fit(self, X_views):
        # Concatenate views
        X_concat = np.hstack(X_views)
        
        # Normalize
        X_concat = normalize(X_concat, norm='l2', axis=1)
        
        # K-Means
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state
        )
        self.labels_ = kmeans.fit_predict(X_concat)
        
        return self


class ConcatSpectral(BaseMultiViewClustering):
    """
    Concatenation + Spectral Clustering baseline.
    """
    
    def __init__(self, n_clusters, n_neighbors=10, random_state=None):
        super().__init__(n_clusters, random_state)
        self.n_neighbors = n_neighbors
        self.name = 'Concat+Spectral'
    
    def fit(self, X_views):
        # Concatenate views
        X_concat = np.hstack(X_views)
        
        # Normalize
        X_concat = normalize(X_concat, norm='l2', axis=1)
        
        # Spectral Clustering
        spectral = SpectralClustering(
            n_clusters=self.n_clusters,
            n_neighbors=self.n_neighbors,
            affinity='nearest_neighbors',
            random_state=self.random_state
        )
        self.labels_ = spectral.fit_predict(X_concat)
        
        return self


class BestViewSpectral(BaseMultiViewClustering):
    """
    Best Single View Spectral Clustering.
    
    Reports the best result among all single views.
    """
    
    def __init__(self, n_clusters, n_neighbors=10, random_state=None):
        super().__init__(n_clusters, random_state)
        self.n_neighbors = n_neighbors
        self.name = 'Best-View-Spectral'
        self.best_view_ = None
    
    def fit(self, X_views, y_true=None):
        """
        Parameters
        ----------
        y_true : array-like, optional
            If provided, select best view based on this.
            Otherwise, use the largest view.
        """
        best_labels = None
        best_score = -1
        
        for v, X in enumerate(X_views):
            X_norm = normalize(X, norm='l2', axis=1)
            
            spectral = SpectralClustering(
                n_clusters=self.n_clusters,
                n_neighbors=min(self.n_neighbors, X.shape[0] - 1),
                affinity='nearest_neighbors',
                random_state=self.random_state
            )
            labels = spectral.fit_predict(X_norm)
            
            if y_true is not None:
                # Use NMI to select best view
                from sklearn.metrics import normalized_mutual_info_score
                score = normalized_mutual_info_score(y_true, labels)
            else:
                # Use silhouette-like heuristic
                score = X.shape[1]  # Prefer larger dimension
            
            if score > best_score:
                best_score = score
                best_labels = labels
                self.best_view_ = v
        
        self.labels_ = best_labels
        return self


class CoRegSpectral(BaseMultiViewClustering):
    """
    Co-Regularized Multi-View Spectral Clustering.
    
    Based on: "Co-regularized Multi-view Spectral Clustering" (NIPS 2011)
    
    Learns view-specific spectral embeddings that are encouraged to agree.
    """
    
    def __init__(self, n_clusters, lambda_reg=0.01, n_neighbors=10, 
                 max_iter=20, random_state=None):
        super().__init__(n_clusters, random_state)
        self.lambda_reg = lambda_reg
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.name = 'Co-Reg Spectral'
    
    def fit(self, X_views):
        n_views = len(X_views)
        n_samples = X_views[0].shape[0]
        K = self.n_clusters
        
        # Build affinity matrices for each view
        W_list = []
        for X in X_views:
            W = self._build_affinity(X)
            W_list.append(W)
        
        # Compute normalized Laplacians
        L_list = []
        for W in W_list:
            D = np.diag(W.sum(axis=1))
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(W.sum(axis=1), 1e-10)))
            L_sym = np.eye(n_samples) - D_inv_sqrt @ W @ D_inv_sqrt
            L_list.append(L_sym)
        
        # Initialize embeddings
        U_list = []
        for L in L_list:
            eigenvalues, eigenvectors = eigh(L)
            U = eigenvectors[:, :K]
            U_list.append(U)
        
        # Co-regularization iterations
        for iteration in range(self.max_iter):
            for v in range(n_views):
                # Compute regularization term from other views
                reg_term = np.zeros((n_samples, n_samples))
                for u in range(n_views):
                    if u != v:
                        reg_term += U_list[u] @ U_list[u].T
                
                # Modified Laplacian
                L_modified = L_list[v] + self.lambda_reg * (np.eye(n_samples) - reg_term / (n_views - 1))
                
                # Update embedding
                eigenvalues, eigenvectors = eigh(L_modified)
                U_list[v] = eigenvectors[:, :K]
        
        # Average embeddings
        U_avg = np.mean(U_list, axis=0)
        U_avg = normalize(U_avg, norm='l2', axis=1)
        
        # K-Means
        kmeans = KMeans(n_clusters=K, n_init=10, random_state=self.random_state)
        self.labels_ = kmeans.fit_predict(U_avg)
        
        return self
    
    def _build_affinity(self, X):
        n_samples = X.shape[0]
        k = min(self.n_neighbors, n_samples - 1)
        
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # RBF kernel with adaptive bandwidth
        sigma = np.median(distances[:, 1:])
        
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j_idx, dist in zip(indices[i, 1:], distances[i, 1:]):
                W[i, j_idx] = np.exp(-dist**2 / (2 * sigma**2))
        
        W = (W + W.T) / 2
        return W


class MultiViewKMeans(BaseMultiViewClustering):
    """
    Multi-View K-Means Clustering.
    
    Joint K-Means across views with view-specific centroids.
    """
    
    def __init__(self, n_clusters, max_iter=100, random_state=None):
        super().__init__(n_clusters, random_state)
        self.max_iter = max_iter
        self.name = 'Multi-View KMeans'
    
    def fit(self, X_views):
        n_views = len(X_views)
        n_samples = X_views[0].shape[0]
        K = self.n_clusters
        
        # Normalize views
        X_views = [normalize(X, norm='l2', axis=1) for X in X_views]
        
        # Initialize labels using concatenation
        X_concat = np.hstack(X_views)
        kmeans = KMeans(n_clusters=K, n_init=1, random_state=self.random_state)
        labels = kmeans.fit_predict(X_concat)
        
        # Alternating optimization
        for iteration in range(self.max_iter):
            old_labels = labels.copy()
            
            # Update centroids for each view
            centroids = []
            for X in X_views:
                C = np.zeros((K, X.shape[1]))
                for k in range(K):
                    mask = labels == k
                    if mask.sum() > 0:
                        C[k] = X[mask].mean(axis=0)
                centroids.append(C)
            
            # Update labels
            distances = np.zeros((n_samples, K))
            for X, C in zip(X_views, centroids):
                for k in range(K):
                    distances[:, k] += np.sum((X - C[k])**2, axis=1)
            
            labels = np.argmin(distances, axis=1)
            
            # Check convergence
            if np.array_equal(labels, old_labels):
                break
        
        self.labels_ = labels
        return self


class MVSC(BaseMultiViewClustering):
    """
    Multi-View Spectral Clustering via bipartite graph.
    
    Simplified implementation of multi-view spectral clustering.
    """
    
    def __init__(self, n_clusters, n_neighbors=10, random_state=None):
        super().__init__(n_clusters, random_state)
        self.n_neighbors = n_neighbors
        self.name = 'MVSC'
    
    def fit(self, X_views):
        n_views = len(X_views)
        n_samples = X_views[0].shape[0]
        K = self.n_clusters
        
        # Build affinity matrices
        W_list = []
        for X in X_views:
            W = self._build_affinity(X)
            W_list.append(W)
        
        # Average affinity matrix
        W_avg = np.mean(W_list, axis=0)
        
        # Spectral clustering on average
        D = np.diag(W_avg.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(W_avg.sum(axis=1), 1e-10)))
        L_sym = np.eye(n_samples) - D_inv_sqrt @ W_avg @ D_inv_sqrt
        
        eigenvalues, eigenvectors = eigh(L_sym)
        U = eigenvectors[:, :K]
        U = normalize(U, norm='l2', axis=1)
        
        kmeans = KMeans(n_clusters=K, n_init=10, random_state=self.random_state)
        self.labels_ = kmeans.fit_predict(U)
        
        return self
    
    def _build_affinity(self, X):
        n_samples = X.shape[0]
        k = min(self.n_neighbors, n_samples - 1)
        
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        sigma = np.median(distances[:, 1:])
        
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j_idx, dist in zip(indices[i, 1:], distances[i, 1:]):
                W[i, j_idx] = np.exp(-dist**2 / (2 * sigma**2))
        
        W = (W + W.T) / 2
        return W


class LMvSC(BaseMultiViewClustering):
    """
    Large-scale Multi-View Subspace Clustering.
    
    Efficient multi-view clustering using anchor graphs.
    """
    
    def __init__(self, n_clusters, n_anchors=100, random_state=None):
        super().__init__(n_clusters, random_state)
        self.n_anchors = n_anchors
        self.name = 'LMvSC'
    
    def fit(self, X_views):
        n_views = len(X_views)
        n_samples = X_views[0].shape[0]
        K = self.n_clusters
        M = min(self.n_anchors, n_samples // 2)
        
        # Sample anchors
        if self.random_state is not None:
            np.random.seed(self.random_state)
        anchor_indices = np.random.choice(n_samples, M, replace=False)
        
        # Build anchor graphs for each view
        Z_list = []
        for X in X_views:
            X_anchors = X[anchor_indices]
            Z = self._build_anchor_graph(X, X_anchors)
            Z_list.append(Z)
        
        # Average anchor graph
        Z_avg = np.mean(Z_list, axis=0)
        
        # Bipartite spectral clustering
        # Compute Z * Z^T approximation for spectral embedding
        ZZt = Z_avg @ Z_avg.T
        
        # Use Nyström-like approach
        D = np.diag(ZZt.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(ZZt.sum(axis=1), 1e-10)))
        L_sym = np.eye(n_samples) - D_inv_sqrt @ ZZt @ D_inv_sqrt
        
        eigenvalues, eigenvectors = eigh(L_sym)
        U = eigenvectors[:, :K]
        U = normalize(U, norm='l2', axis=1)
        
        kmeans = KMeans(n_clusters=K, n_init=10, random_state=self.random_state)
        self.labels_ = kmeans.fit_predict(U)
        
        return self
    
    def _build_anchor_graph(self, X, X_anchors):
        n_samples = X.shape[0]
        M = X_anchors.shape[0]
        
        # Compute distances to anchors
        distances = np.zeros((n_samples, M))
        for m in range(M):
            distances[:, m] = np.sum((X - X_anchors[m])**2, axis=1)
        
        # Find k nearest anchors
        k = min(5, M)
        Z = np.zeros((n_samples, M))
        
        for i in range(n_samples):
            nearest = np.argsort(distances[i])[:k]
            sigma = distances[i, nearest[-1]] + 1e-10
            weights = np.exp(-distances[i, nearest] / sigma)
            weights /= weights.sum()
            Z[i, nearest] = weights
        
        return Z


class MLAN(BaseMultiViewClustering):
    """
    Multi-View Learning with Adaptive Neighbors.
    
    Learns view-specific neighbor graphs and fuses them adaptively.
    """
    
    def __init__(self, n_clusters, n_neighbors=10, lambda_reg=1.0, 
                 max_iter=20, random_state=None):
        super().__init__(n_clusters, random_state)
        self.n_neighbors = n_neighbors
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.name = 'MLAN'
    
    def fit(self, X_views):
        n_views = len(X_views)
        n_samples = X_views[0].shape[0]
        K = self.n_clusters
        
        # Initialize view weights
        weights = np.ones(n_views) / n_views
        
        # Build initial similarity matrices
        S_list = []
        for X in X_views:
            S = self._build_similarity(X)
            S_list.append(S)
        
        for iteration in range(self.max_iter):
            # Fuse graphs with current weights
            S_fused = np.zeros((n_samples, n_samples))
            for v, S in enumerate(S_list):
                S_fused += weights[v] * S
            
            # Update view weights based on graph quality
            new_weights = np.zeros(n_views)
            for v, S in enumerate(S_list):
                # Measure consistency with fused graph
                consistency = np.trace(S @ S_fused) / (np.linalg.norm(S, 'fro') * np.linalg.norm(S_fused, 'fro') + 1e-10)
                new_weights[v] = np.exp(consistency / self.lambda_reg)
            
            weights = new_weights / (new_weights.sum() + 1e-10)
        
        # Final spectral clustering
        D = np.diag(S_fused.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(S_fused.sum(axis=1), 1e-10)))
        L_sym = np.eye(n_samples) - D_inv_sqrt @ S_fused @ D_inv_sqrt
        
        eigenvalues, eigenvectors = eigh(L_sym)
        U = eigenvectors[:, :K]
        U = normalize(U, norm='l2', axis=1)
        
        kmeans = KMeans(n_clusters=K, n_init=10, random_state=self.random_state)
        self.labels_ = kmeans.fit_predict(U)
        
        self.view_weights_ = weights
        return self
    
    def _build_similarity(self, X):
        n_samples = X.shape[0]
        k = min(self.n_neighbors, n_samples - 1)
        
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        sigma = np.median(distances[:, 1:])
        
        S = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j_idx, dist in zip(indices[i, 1:], distances[i, 1:]):
                S[i, j_idx] = np.exp(-dist**2 / (2 * sigma**2))
        
        S = (S + S.T) / 2
        return S


def get_baseline_methods(n_clusters, random_state=None):
    """
    Get all baseline methods for comparison.
    
    Returns
    -------
    methods : dict
        Dictionary of method name -> method instance.
    """
    return {
        'Concat+KMeans': ConcatKMeans(n_clusters, random_state),
        'Concat+Spectral': ConcatSpectral(n_clusters, random_state=random_state),
        # 'Best-View': BestViewSpectral(n_clusters, random_state=random_state),
        # 'Co-Reg': CoRegSpectral(n_clusters, random_state=random_state),
        'MV-KMeans': MultiViewKMeans(n_clusters, random_state=random_state),
        # 'MVSC': MVSC(n_clusters, random_state=random_state),
        'LMvSC': LMvSC(n_clusters, random_state=random_state),
        # 'MLAN': MLAN(n_clusters, random_state=random_state),
    }
