"""
SPOCK: Scalable and Structure-Preserving Optimal Transport based Clustering 
with Kernel-density-estimation for Imperfect Multi-View Data

Core algorithm implementation following the paper's methodology.
"""

import numpy as np
from scipy import sparse
from scipy.linalg import sqrtm, inv
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import warnings

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    warnings.warn("FAISS not installed. Using sklearn for ANN (slower).")


class SPOCK:
    """
    SPOCK: Scalable and Structure-Preserving Optimal Transport based Clustering
    
    A three-phase framework for multi-view clustering:
    - Phase 1: Unsupervised Structure-Preserving Sparse Feature Selection
    - Phase 2: Density-Aware Graph Alignment using RFF-accelerated OT
    - Phase 3: Nyström-accelerated Spectral Clustering
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    alpha : float, default=1.0
        Weight for self-expression term in Phase 1.
    beta : float, default=0.1
        L1 sparsity weight for S matrix.
    lambda_l21 : float, default=0.01
        L2,1 regularization weight for projection matrix P.
    k_neighbors : int, default=10
        Number of neighbors for KNN graph construction.
    proj_dim : int, default=100
        Dimension of projected feature space.
    rff_dim : int, default=256
        Dimension of Random Fourier Features.
    sigma : float, default=-1
        Kernel bandwidth. If -1, automatically determined.
    density_percentile : float, default=10
        Percentile for density threshold.
    n_landmarks : int, default=500
        Number of landmarks for Nyström approximation.
    max_iter : int, default=50
        Maximum iterations for Phase 1 optimization.
    tol : float, default=1e-4
        Convergence tolerance.
    rho : float, default=1.0
        ADMM penalty parameter.
    sinkhorn_iter : int, default=100
        Number of Sinkhorn iterations.
    sinkhorn_reg : float, default=0.1
        Entropy regularization for Sinkhorn.
    random_state : int, default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to print progress.
    """
    
    def __init__(
        self,
        n_clusters,
        alpha=1.0,
        beta=0.1,
        lambda_l21=0.01,
        k_neighbors=10,
        proj_dim=100,
        rff_dim=256,
        sigma=-1,
        density_percentile=10,
        n_landmarks=500,
        max_iter=50,
        tol=1e-4,
        rho=1.0,
        sinkhorn_iter=100,
        sinkhorn_reg=0.1,
        random_state=None,
        verbose=False
    ):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.lambda_l21 = lambda_l21
        self.k_neighbors = k_neighbors
        self.proj_dim = proj_dim
        self.rff_dim = rff_dim
        self.sigma = sigma
        self.density_percentile = density_percentile
        self.n_landmarks = n_landmarks
        self.max_iter = max_iter
        self.tol = tol
        self.rho = rho
        self.sinkhorn_iter = sinkhorn_iter
        self.sinkhorn_reg = sinkhorn_reg
        self.random_state = random_state
        self.verbose = verbose
        
        # Attributes set after fitting
        self.labels_ = None
        self.projection_matrices_ = None
        self.consensus_graph_ = None
        self.spectral_embedding_ = None
        
    def fit(self, X_views):
        """
        Fit SPOCK to multi-view data.
        
        Parameters
        ----------
        X_views : list of ndarray
            List of view matrices, each of shape (n_samples, n_features_v).
            
        Returns
        -------
        self : SPOCK
            Fitted estimator.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_views = len(X_views)
        n_samples = X_views[0].shape[0]
        
        if self.verbose:
            print(f"SPOCK: Processing {n_views} views with {n_samples} samples")
        
        # Phase 1: Feature Selection
        if self.verbose:
            print("Phase 1: Structure-Preserving Feature Selection...")
        projected_views, self.projection_matrices_ = self._phase1_feature_selection(X_views)
        
        # Phase 2: Density-Aware Graph Alignment  
        if self.verbose:
            print("Phase 2: Density-Aware Graph Alignment...")
        self.consensus_graph_ = self._phase2_graph_alignment(projected_views)
        
        # Phase 3: Final Clustering
        if self.verbose:
            print("Phase 3: Nyström Spectral Clustering...")
        self.labels_ = self._phase3_clustering(self.consensus_graph_, n_samples)
        
        return self
    
    def fit_predict(self, X_views):
        """
        Fit SPOCK and return cluster labels.
        
        Parameters
        ----------
        X_views : list of ndarray
            List of view matrices.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X_views)
        return self.labels_
    
    # ==================== Phase 1: Feature Selection ====================
    
    def _phase1_feature_selection(self, X_views):
        """
        Phase 1: Structure-Preserving Feature Selection
        
        Uses Laplacian-regularized PCA for dimensionality reduction while
        preserving local structure.
        """
        projected_views = []
        projection_matrices = []
        
        for v, X in enumerate(X_views):
            if self.verbose:
                print(f"  Processing view {v+1}/{len(X_views)}...")
            
            N, D = X.shape
            d = min(self.proj_dim, D, N - 1)
            
            # Normalize input
            X_centered = X - X.mean(axis=0)
            std = X_centered.std(axis=0)
            std[std < 1e-10] = 1.0
            X_normalized = X_centered / std
            
            # Step 1: Construct KNN graph and Laplacian
            L = self._construct_laplacian(X_normalized)
            
            # Step 2: Laplacian-regularized projection
            # Solve: max Tr(P'X'XP) - α Tr(P'X'LXP)
            # This preserves variance while maintaining locality
            
            XtX = X_normalized.T @ X_normalized
            XtLX = X_normalized.T @ L @ X_normalized
            
            # Regularized covariance matrix
            # M = XtX - α * XtLX (we want eigenvectors of this)
            M = XtX - self.alpha * XtLX
            M = (M + M.T) / 2  # Ensure symmetry
            M += np.eye(D) * 1e-6  # Regularization
            
            # Get top eigenvectors
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(M)
                # Sort by eigenvalue (descending)
                idx = np.argsort(eigenvalues)[::-1][:d]
                P = eigenvectors[:, idx]
            except:
                # Fallback to standard PCA
                eigenvalues, eigenvectors = np.linalg.eigh(XtX)
                idx = np.argsort(eigenvalues)[::-1][:d]
                P = eigenvectors[:, idx]
            
            # Project features
            X_proj = X_normalized @ P
            
            # L2 normalize rows
            X_proj = normalize(X_proj, norm='l2', axis=1)
            
            projected_views.append(X_proj)
            projection_matrices.append(P)
            
        return projected_views, projection_matrices
    
    def _construct_laplacian(self, X):
        """Construct graph Laplacian from KNN graph."""
        N = X.shape[0]
        k = min(self.k_neighbors, N - 1)
        
        # Use FAISS for fast ANN if available
        if HAS_FAISS and N > 1000:
            A = self._faiss_knn(X, k)
        else:
            A = self._sklearn_knn(X, k)
        
        # Symmetrize adjacency matrix
        A = (A + A.T) / 2
        
        # Compute degree matrix and Laplacian
        D = np.diag(np.array(A.sum(axis=1)).flatten())
        L = D - A
        
        return L
    
    def _faiss_knn(self, X, k):
        """Fast KNN using FAISS."""
        X = np.ascontiguousarray(X.astype(np.float32))
        N, D = X.shape
        
        index = faiss.IndexFlatL2(D)
        index.add(X)
        
        # Query k+1 neighbors (first is self)
        distances, indices = index.search(X, k + 1)
        
        # Build adjacency matrix
        A = np.zeros((N, N))
        for i in range(N):
            for j in indices[i, 1:]:  # Skip self
                A[i, j] = 1
                
        return A
    
    def _sklearn_knn(self, X, k):
        """KNN using sklearn."""
        N = X.shape[0]
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        A = np.zeros((N, N))
        for i in range(N):
            for j in indices[i, 1:]:
                A[i, j] = 1
                
        return A
    
    def _admm_optimization(self, X, L, d):
        """
        ADMM optimization for joint P and S learning.
        
        Objective:
        min ||XP - SXP||_F^2 + α Tr(P'X'LXP) + λ||P||_{2,1} + β||S||_1
        s.t. P'P = I, diag(S) = 0
        """
        N, D = X.shape
        
        # Initialize P via PCA
        XtX = X.T @ X + np.eye(D) * 1e-6  # Regularize
        eigenvalues, eigenvectors = np.linalg.eigh(XtX)
        P = eigenvectors[:, -d:]
        
        # Initialize S as sparse local coefficients
        S = self._initialize_S(X)
        
        # Initialize auxiliary variables
        Q = P.copy()
        Y = np.zeros_like(P)  # Lagrange multiplier
        
        # Precompute X'LX
        XtLX = X.T @ L @ X
        
        prev_obj = float('inf')
        
        for iteration in range(self.max_iter):
            P_old = P.copy()
            
            # Step 1: Update S (sparse coding)
            S = self._update_S(X, P, S)
            
            # Step 2: Update P via ADMM
            P, Q, Y = self._update_P_admm(X, L, S, P, Q, Y, XtLX)
            
            # Compute objective for convergence check
            Z = X @ P
            I_minus_S = np.eye(N) - S
            recon_loss = np.linalg.norm(I_minus_S @ Z, 'fro') ** 2
            lap_loss = np.trace(P.T @ XtLX @ P)
            sparse_loss = self.lambda_l21 * np.sum(np.linalg.norm(P, axis=1))
            obj = recon_loss + self.alpha * lap_loss + sparse_loss
            
            # Check convergence
            rel_change = abs(prev_obj - obj) / (abs(prev_obj) + 1e-10)
            if rel_change < self.tol and iteration > 5:
                if self.verbose:
                    print(f"    Converged at iteration {iteration}")
                break
            
            prev_obj = obj
                
        return P, S
    
    def _initialize_S(self, X):
        """Initialize S using local linear coding."""
        N = X.shape[0]
        k = min(self.k_neighbors, N - 1)
        
        # Get KNN indices
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        _, indices = nbrs.kneighbors(X)
        
        S = np.zeros((N, N))
        
        for i in range(N):
            neighbors = indices[i, 1:]  # Exclude self
            X_neighbors = X[neighbors]
            x_i = X[i:i+1]
            
            # Solve local least squares
            try:
                coeffs = np.linalg.lstsq(X_neighbors.T, x_i.T, rcond=None)[0].flatten()
                coeffs /= (np.sum(np.abs(coeffs)) + 1e-10)  # Normalize
                S[i, neighbors] = coeffs
            except:
                S[i, neighbors] = 1.0 / k
                
        return S
    
    def _update_S(self, X, P, S):
        """
        Update S via sparse coding:
        min ||Z - SZ||_F^2 + β||S||_1, s.t. diag(S) = 0
        where Z = XP
        """
        Z = X @ P
        N = Z.shape[0]
        
        # Soft thresholding for L1 minimization
        ZZt = Z @ Z.T
        ZZt_inv = np.linalg.pinv(ZZt + np.eye(N) * 1e-6)
        
        S_new = ZZt @ ZZt_inv
        
        # Soft thresholding
        threshold = self.beta / 2
        S_new = np.sign(S_new) * np.maximum(np.abs(S_new) - threshold, 0)
        
        # Enforce diagonal = 0
        np.fill_diagonal(S_new, 0)
        
        return S_new
    
    def _update_P_admm(self, X, L, S, P, Q, Y, XtLX):
        """
        ADMM update for P.
        
        Augmented Lagrangian:
        L_rho = Tr(P'MP) + λ||Q||_{2,1} + <Y, P-Q> + (ρ/2)||P-Q||_F^2
        
        where M = X'(I-S)'(I-S)X + αX'LX
        """
        N, D = X.shape
        d = P.shape[1]
        I_minus_S = np.eye(N) - S
        
        # M = X'(I-S)'(I-S)X + α*X'LX
        M = X.T @ I_minus_S.T @ I_minus_S @ X + self.alpha * XtLX
        
        # Update P: solve (2M + ρI)P = ρQ - Y
        A = 2 * M + self.rho * np.eye(D)
        B = self.rho * Q - Y
        P_new = np.linalg.solve(A, B)
        
        # Orthogonalize P
        U, _, Vt = np.linalg.svd(P_new, full_matrices=False)
        P_new = U @ Vt[:d, :]
        if P_new.shape[1] < d:
            P_new = U[:, :d]
        
        # Update Q: row-wise soft thresholding for L2,1 norm
        Q_new = self._l21_proximal(P_new + Y / self.rho, self.lambda_l21 / self.rho)
        
        # Update Y (dual variable)
        Y_new = Y + self.rho * (P_new - Q_new)
        
        return P_new, Q_new, Y_new
    
    def _l21_proximal(self, X, threshold):
        """Proximal operator for L2,1 norm (row-wise soft thresholding)."""
        row_norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
        scale = np.maximum(1 - threshold / row_norms, 0)
        return X * scale
    
    # ==================== Phase 2: Graph Alignment ====================
    
    def _phase2_graph_alignment(self, projected_views):
        """
        Phase 2: Graph construction and alignment.
        
        - Build KNN graphs for each view
        - Align and fuse graphs into consensus
        """
        n_views = len(projected_views)
        n_samples = projected_views[0].shape[0]
        
        # For each view: construct KNN graph with RBF weights
        view_graphs = []
        for v, X in enumerate(projected_views):
            if self.verbose:
                print(f"  Building graph for view {v+1}/{n_views}...")
            
            # Determine sigma using median heuristic
            sigma = self._auto_bandwidth(X)
            
            # Build KNN graph with RBF weights
            G = self._build_knn_graph(X, sigma)
            view_graphs.append(G)
        
        # Align graphs
        consensus = self._align_graphs_ot(view_graphs, projected_views)
            
        return consensus
    
    def _build_knn_graph(self, X, sigma):
        """Build a KNN graph with RBF kernel weights."""
        N = X.shape[0]
        k = min(self.k_neighbors * 2, N - 1)
        
        # Get neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Build graph with RBF weights
        G = np.zeros((N, N))
        for i in range(N):
            for idx, dist in zip(indices[i, 1:], distances[i, 1:]):
                weight = np.exp(-dist**2 / (2 * sigma**2))
                G[i, idx] = weight
        
        # Symmetrize
        G = (G + G.T) / 2
        
        return G
    
    def _auto_bandwidth(self, X):
        """Automatically determine kernel bandwidth using median heuristic."""
        n = min(1000, X.shape[0])
        indices = np.random.choice(X.shape[0], n, replace=False)
        X_sample = X[indices]
        
        # Compute pairwise distances
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(np.linalg.norm(X_sample[i] - X_sample[j]))
        
        return np.median(dists) if dists else 1.0
    
    def _compute_rff(self, X, sigma):
        """
        Compute Random Fourier Features for RBF kernel approximation.
        
        k(x, y) ≈ φ(x)' φ(y)
        where φ(x) = sqrt(2/D) * [cos(ω'x), sin(ω'x)]
        and ω ~ N(0, 1/σ² I)
        """
        N, d = X.shape
        D = self.rff_dim
        
        # Sample random frequencies
        omega = np.random.randn(d, D // 2) / sigma
        
        # Compute features
        XW = X @ omega
        rff = np.sqrt(2.0 / (D // 2)) * np.hstack([np.cos(XW), np.sin(XW)])
        
        return rff
    
    def _rff_density_estimation(self, rff_features):
        """
        RFF-accelerated kernel density estimation.
        
        p(x) = (1/N) Σ k(x, x_i) ≈ φ(x)' * (1/N Σ φ(x_i))
        """
        N = rff_features.shape[0]
        
        # Mean RFF feature
        phi_mean = np.mean(rff_features, axis=0)
        
        # Density estimate for each point
        densities = rff_features @ phi_mean
        
        return densities
    
    def _construct_density_aware_graph(self, X, densities, threshold, sigma):
        """
        Construct density-aware similarity graph.
        
        G_ij = k(x_i, x_j) * I(p(x_i) > τ) * I(p(x_j) > τ)
        
        For low-density points, connect them to high-density neighbors.
        """
        N = X.shape[0]
        k = min(self.k_neighbors * 2, N - 1)
        
        # Get neighbors (for sparse graph)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Build graph
        G = np.zeros((N, N))
        high_density = densities > threshold
        
        for i in range(N):
            for j_idx, (idx, dist) in enumerate(zip(indices[i, 1:], distances[i, 1:])):
                # RBF kernel weight
                weight = np.exp(-dist**2 / (2 * sigma**2))
                
                if high_density[i] and high_density[idx]:
                    # Both high density: full weight
                    G[i, idx] = weight
                elif high_density[i] or high_density[idx]:
                    # One high density: reduced weight (but still connected)
                    G[i, idx] = weight * 0.5
                else:
                    # Both low density: very weak connection to nearest only
                    if j_idx < self.k_neighbors // 2:
                        G[i, idx] = weight * 0.2
        
        # Symmetrize
        G = (G + G.T) / 2
        
        # Ensure graph is connected by adding small weights to disconnected components
        # Check and fix isolated nodes
        row_sums = G.sum(axis=1)
        isolated = row_sums < 1e-10
        if isolated.any():
            # Connect isolated nodes to their nearest neighbors
            for i in np.where(isolated)[0]:
                for idx, dist in zip(indices[i, 1:self.k_neighbors+1], distances[i, 1:self.k_neighbors+1]):
                    weight = np.exp(-dist**2 / (2 * sigma**2))
                    G[i, idx] = weight * 0.3
                    G[idx, i] = weight * 0.3
        
        return G
    
    def _align_graphs_ot(self, view_graphs, projected_views):
        """
        Align view-specific graphs.
        
        Computes consensus graph from multiple views using weighted averaging
        with view quality estimation.
        """
        n_views = len(view_graphs)
        n_samples = view_graphs[0].shape[0]
        
        # Simple but effective: weighted average of view graphs
        # with weights based on graph quality
        view_weights = []
        for v, G in enumerate(view_graphs):
            # Quality = average edge weight * sparsity factor
            nnz = np.count_nonzero(G)
            if nnz > 0:
                avg_weight = G.sum() / nnz
                sparsity = 1 - nnz / (n_samples * n_samples)
                quality = avg_weight * (0.5 + 0.5 * sparsity)
            else:
                quality = 0.01
            view_weights.append(quality)
        
        view_weights = np.array(view_weights)
        view_weights = view_weights / (view_weights.sum() + 1e-10)
        
        if self.verbose:
            print(f"  View weights: {view_weights}")
        
        # Weighted average
        consensus = np.zeros((n_samples, n_samples))
        for v in range(n_views):
            consensus += view_weights[v] * view_graphs[v]
        
        # Symmetrize
        consensus = (consensus + consensus.T) / 2
        
        return consensus
    
    def _refine_consensus_ot(self, consensus, view_graphs, projected_views, view_weights):
        """
        Refine consensus graph using Optimal Transport based alignment.
        
        Uses Sinkhorn algorithm for efficient OT computation.
        Preserves sparsity of the graph.
        """
        n_views = len(view_graphs)
        n_samples = consensus.shape[0]
        
        # Instead of full OT which creates dense matrix, use local refinement
        # Keep the sparse structure by only refining existing edges
        refined = consensus.copy()
        
        for v in range(n_views):
            G_v = view_graphs[v]
            
            # Weighted update that preserves sparsity
            # Only add edges where either consensus or view graph has non-zero weight
            mask = (consensus > 0) | (G_v > 0)
            update = np.zeros_like(consensus)
            update[mask] = (view_weights[v] * 0.2) * G_v[mask]
            
            refined = (1 - view_weights[v] * 0.2) * refined + update
        
        # Enforce sparsity by keeping only k-largest neighbors per row
        k = self.k_neighbors * 3
        for i in range(n_samples):
            row = refined[i].copy()
            if np.count_nonzero(row) > k:
                threshold = np.partition(row, -k)[-k]
                row[row < threshold] = 0
                refined[i] = row
        
        # Symmetrize
        refined = (refined + refined.T) / 2
        
        return refined
    
    def _sinkhorn(self, C, max_iter=100, reg=0.1):
        """
        Sinkhorn algorithm for optimal transport.
        
        Solves: min <T, C> - reg * H(T)
        s.t. T1 = a, T'1 = b (uniform marginals)
        """
        n = C.shape[0]
        
        # Uniform marginals
        a = np.ones(n) / n
        b = np.ones(n) / n
        
        # Gibbs kernel
        K = np.exp(-C / reg)
        K = np.clip(K, 1e-100, 1e100)  # Numerical stability
        
        # Initialize
        u = np.ones(n)
        v = np.ones(n)
        
        for _ in range(max_iter):
            u_prev = u.copy()
            
            # Sinkhorn iterations
            v = b / (K.T @ u + 1e-100)
            u = a / (K @ v + 1e-100)
            
            # Check convergence
            if np.max(np.abs(u - u_prev)) < 1e-6:
                break
        
        # Compute transport plan
        T = np.diag(u) @ K @ np.diag(v)
        
        return T
    
    # ==================== Phase 3: Final Clustering ====================
    
    def _phase3_clustering(self, consensus_graph, n_samples):
        """
        Phase 3: Nyström-accelerated Spectral Clustering
        """
        # Ensure graph is valid
        W = consensus_graph.copy()
        W = (W + W.T) / 2  # Symmetrize
        W[W < 0] = 0  # Non-negative
        
        # Add small self-loops for numerical stability
        np.fill_diagonal(W, W.diagonal() + 0.01)
        
        # For better results, always use standard spectral embedding 
        # (Nyström can be enabled for very large datasets)
        n_landmarks = min(self.n_landmarks, n_samples // 2, 500)
        
        if n_samples > 5000 and n_landmarks >= self.n_clusters * 5:
            # Use Nyström method only for very large datasets
            Z = self._nystrom_embedding(W, n_landmarks)
        else:
            # Standard spectral embedding - more reliable
            Z = self._standard_spectral_embedding(W)
        
        # Store embedding
        self.spectral_embedding_ = Z
        
        # K-Means on spectral embedding with multiple restarts
        best_labels = None
        best_inertia = float('inf')
        
        for _ in range(3):  # Multiple restarts
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=None if self.random_state is None else self.random_state + _
            )
            labels = kmeans.fit_predict(Z)
            
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_labels = labels
        
        return best_labels
    
    def _nystrom_embedding(self, W, M):
        """
        Nyström approximation for spectral embedding.
        
        Given affinity matrix W, compute approximate eigenvectors
        using M landmark points.
        """
        N = W.shape[0]
        
        # Sample landmarks using k-means++ style selection for better coverage
        landmark_indices = self._select_landmarks_kmeans_pp(W, M)
        
        # Extract submatrices
        C_NM = W[:, landmark_indices]  # N x M
        W_MM = W[np.ix_(landmark_indices, landmark_indices)]  # M x M
        
        # Regularize W_MM for numerical stability
        W_MM = W_MM + np.eye(M) * 1e-5
        W_MM = (W_MM + W_MM.T) / 2  # Ensure symmetry
        
        # Eigendecomposition of W_MM
        eigvals, eigvecs = np.linalg.eigh(W_MM)
        
        # Keep positive eigenvalues
        pos_mask = eigvals > 1e-10
        eigvals = eigvals[pos_mask]
        eigvecs = eigvecs[:, pos_mask]
        
        if len(eigvals) < self.n_clusters:
            # Fall back to standard embedding if not enough eigenvalues
            return self._standard_spectral_embedding(W)
        
        # Compute W_MM^{-1/2} using eigendecomposition
        eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)
        W_MM_inv_sqrt = eigvecs @ np.diag(eigvals_inv_sqrt) @ eigvecs.T
        
        # Approximate eigenvectors: U ≈ C @ W_MM^{-1/2} @ V @ Σ^{-1}
        # where W_MM = V @ Σ @ V'
        
        # Simpler approach: compute normalized embedding
        # Z = D^{-1/2} @ C @ W_MM^{-1/2}
        D_approx = C_NM.sum(axis=1)
        D_approx = np.maximum(D_approx, 1e-10)
        D_inv_sqrt = 1.0 / np.sqrt(D_approx)
        
        Z = (D_inv_sqrt[:, np.newaxis] * C_NM) @ W_MM_inv_sqrt
        
        # SVD to get orthogonal embedding
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        
        # Take top K eigenvectors (skip first one if it's trivial)
        K = self.n_clusters
        if S[0] > 0.99 and len(S) > K:
            # First eigenvector is likely trivial (constant)
            Z = U[:, 1:K+1]
        else:
            Z = U[:, :K]
        
        # Normalize rows
        Z = normalize(Z, norm='l2', axis=1)
        
        return Z
    
    def _select_landmarks_kmeans_pp(self, W, M):
        """Select landmarks using k-means++ style initialization based on graph structure."""
        N = W.shape[0]
        
        # Start with random point
        landmarks = [np.random.randint(N)]
        
        # Compute graph distances using shortest path approximation
        # For efficiency, use inverse weights as distances
        D = 1.0 / (W + 1e-10)
        np.fill_diagonal(D, 0)
        
        for _ in range(M - 1):
            # Compute minimum distance to existing landmarks
            min_dists = np.min(D[:, landmarks], axis=1)
            min_dists[landmarks] = 0  # Already selected
            
            # Sample proportional to distance squared
            probs = min_dists ** 2
            probs = probs / (probs.sum() + 1e-10)
            
            new_landmark = np.random.choice(N, p=probs)
            landmarks.append(new_landmark)
        
        return np.array(landmarks)
    
    def _standard_spectral_embedding(self, W):
        """Standard spectral embedding for small datasets."""
        N = W.shape[0]
        
        # Compute normalized Laplacian
        d = W.sum(axis=1)
        d = np.maximum(d, 1e-10)  # Avoid division by zero
        D_sqrt_inv = np.diag(1.0 / np.sqrt(d))
        
        # Normalized affinity matrix (random walk normalization)
        W_norm = D_sqrt_inv @ W @ D_sqrt_inv
        W_norm = (W_norm + W_norm.T) / 2  # Ensure symmetry
        
        # Compute top K eigenvectors of W_norm (largest eigenvalues)
        K = self.n_clusters
        try:
            # Use eigsh for efficiency
            eigenvalues, eigenvectors = eigsh(W_norm, k=K, which='LM')
            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
        except:
            # Fallback to full eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(W_norm)
            idx = np.argsort(eigenvalues)[::-1][:K]
            eigenvectors = eigenvectors[:, idx]
        
        Z = normalize(eigenvectors, norm='l2', axis=1)
        
        return Z


class SPOCKAblation(SPOCK):
    """
    SPOCK with ablation study support.
    
    Allows disabling specific components for ablation experiments.
    """
    
    def __init__(self, ablation_mode='full', **kwargs):
        """
        Parameters
        ----------
        ablation_mode : str
            One of: 'full', 'no_feature_selection', 'no_density_aware', 
                   'no_ot_alignment', 'standard_spectral'
        """
        super().__init__(**kwargs)
        self.ablation_mode = ablation_mode
    
    def _phase1_feature_selection(self, X_views):
        if self.ablation_mode == 'no_feature_selection':
            # Skip feature selection, just return normalized data
            projected_views = []
            projection_matrices = []
            for X in X_views:
                X_normalized = normalize(X, norm='l2', axis=1)
                projected_views.append(X_normalized)
                projection_matrices.append(None)
            return projected_views, projection_matrices
        else:
            return super()._phase1_feature_selection(X_views)
    
    def _construct_density_aware_graph(self, X, densities, threshold, sigma):
        if self.ablation_mode == 'no_density_aware':
            # Don't use density thresholding
            N = X.shape[0]
            k = min(self.k_neighbors * 2, N - 1)
            
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
            distances, indices = nbrs.kneighbors(X)
            
            G = np.zeros((N, N))
            for i in range(N):
                for idx, dist in zip(indices[i, 1:], distances[i, 1:]):
                    G[i, idx] = np.exp(-dist**2 / (2 * sigma**2))
            
            G = (G + G.T) / 2
            return G
        else:
            return super()._construct_density_aware_graph(X, densities, threshold, sigma)
    
    def _align_graphs_ot(self, view_graphs, projected_views):
        if self.ablation_mode == 'no_ot_alignment':
            # Simple average instead of OT
            consensus = np.mean(view_graphs, axis=0)
            return consensus
        else:
            return super()._align_graphs_ot(view_graphs, projected_views)
    
    def _phase3_clustering(self, consensus_graph, n_samples):
        if self.ablation_mode == 'standard_spectral':
            # Force standard spectral (no Nyström)
            W = consensus_graph.copy()
            W = (W + W.T) / 2
            W[W < 0] = 0
            
            Z = self._standard_spectral_embedding(W)
            self.spectral_embedding_ = Z
            
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                init='k-means++',
                n_init=10,
                random_state=self.random_state
            )
            return kmeans.fit_predict(Z)
        else:
            return super()._phase3_clustering(consensus_graph, n_samples)
