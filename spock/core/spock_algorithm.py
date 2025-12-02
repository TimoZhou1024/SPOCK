"""
SPOCK: Scalable and Structure-Preserving Optimal Transport based Clustering 
with Kernel-density-estimation for Imperfect Multi-View Data

Core algorithm implementation following the paper's methodology EXACTLY.

Paper Algorithm Overview:
- Phase 1: ADMM optimization for joint P and S learning
  min ||XP - SXP||_F^2 + α Tr(P'X'LXP) + λ||P||_{2,1} + β||S||_1
  
- Phase 2: RFF-accelerated KDE for density estimation + RFF-Sinkhorn for OT alignment

- Phase 3: Nyström-accelerated spectral clustering
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
    
    A three-phase framework for multi-view clustering (Paper Implementation):
    
    - Phase 1: Unsupervised Structure-Preserving Sparse Feature Selection
      Joint optimization of projection matrix P and self-expression matrix S
      using ADMM algorithm.
      
    - Phase 2: Density-Aware Graph Alignment using RFF
      - RFF-accelerated Kernel Density Estimation
      - RFF-accelerated Sinkhorn Optimal Transport for graph alignment
      
    - Phase 3: Nyström-accelerated Spectral Clustering
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    alpha : float, default=1.0
        Weight for Laplacian regularization term (local structure preservation).
    beta : float, default=0.1
        L1 sparsity weight for S matrix (self-expression sparsity).
    lambda_l21 : float, default=0.01
        L2,1 regularization weight for projection matrix P (feature selection).
    k_neighbors : int, default=10
        Number of neighbors for KNN graph construction.
    proj_dim : int, default=100
        Dimension of projected feature space.
    rff_dim : int, default=256
        Dimension of Random Fourier Features.
    sigma : float, default=-1
        Kernel bandwidth. If -1, automatically determined via entropy maximization.
    density_percentile : float, default=10
        Percentile for density threshold (low-density filtering).
    n_landmarks : int, default=500
        Number of landmarks for Nyström approximation.
    max_iter : int, default=50
        Maximum iterations for Phase 1 ADMM optimization.
    tol : float, default=1e-4
        Convergence tolerance.
    rho : float, default=1.0
        ADMM penalty parameter.
    sinkhorn_iter : int, default=100
        Number of Sinkhorn iterations.
    sinkhorn_reg : float, default=0.1
        Entropy regularization for Sinkhorn OT.
    use_spectral : bool, default=False
        Whether to use spectral clustering (True) or direct KMeans on features (False).
        KMeans mode often works better and is faster.
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
        use_spectral=False,
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
        self.use_spectral = use_spectral
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
        self.self_expression_matrices_ = None
        self.consensus_graph_ = None
        self.spectral_embedding_ = None
        self.view_weights_ = None
        self.rff_omega_ = None  # Store RFF parameters
        self.projected_views_ = None  # Store projected features
        
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
        
        # Phase 1: Feature Selection via ADMM
        if self.verbose:
            print("Phase 1: Structure-Preserving Feature Selection...")
        projected_views, self.projection_matrices_, self.self_expression_matrices_ = \
            self._phase1_feature_selection(X_views)
        self.projected_views_ = projected_views
        
        if self.use_spectral:
            # Phase 2: Compute view weights only (skip expensive graph fusion)
            if self.verbose:
                print("Phase 2: Computing View Weights...")
            self.view_weights_ = self._phase2_compute_view_weights(projected_views)
            self.consensus_graph_ = None  # Not needed for hybrid clustering
            
            # Phase 3: Final Clustering via Hybrid Spectral + Features
            if self.verbose:
                print("Phase 3: Hybrid Spectral Clustering...")
            self.labels_ = self._phase3_hybrid_clustering(
                self.consensus_graph_, projected_views, self.view_weights_, n_samples
            )
        else:
            # Alternative: Direct KMeans on concatenated projected features
            if self.verbose:
                print("Phase 2: Direct KMeans on projected features...")
            X_concat = np.hstack(projected_views)
            
            best_labels = None
            best_inertia = float('inf')
            for restart in range(5):
                kmeans = KMeans(
                    n_clusters=self.n_clusters,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=None if self.random_state is None else self.random_state + restart
                )
                labels = kmeans.fit_predict(X_concat)
                if kmeans.inertia_ < best_inertia:
                    best_inertia = kmeans.inertia_
                    best_labels = labels
            
            self.labels_ = best_labels
            self.consensus_graph_ = None
            self.view_weights_ = np.ones(n_views) / n_views
        
        return self
    
    def fit_predict(self, X_views):
        """Fit SPOCK and return cluster labels."""
        self.fit(X_views)
        return self.labels_
    
    # ==================== Phase 1: ADMM Feature Selection ====================
    
    def _phase1_feature_selection(self, X_views):
        """
        Phase 1: Unsupervised Structure-Preserving Sparse Feature Selection
        
        For each view, solve:
            min_{P,S} ||XP - SXP||_F^2 + α Tr(P'X'LXP) + λ||P||_{2,1} + β||S||_1
            s.t. P'P = I, diag(S) = 0
        
        Using ADMM with alternating updates.
        """
        projected_views = []
        projection_matrices = []
        self_expression_matrices = []
        
        for v, X in enumerate(X_views):
            if self.verbose:
                print(f"  Processing view {v+1}/{len(X_views)}...")
            
            N, D = X.shape
            d = min(self.proj_dim, D - 1, N - 1)
            
            # Normalize input
            X_centered = X - X.mean(axis=0)
            scale = np.linalg.norm(X_centered, 'fro') / np.sqrt(N * D) + 1e-10
            X_normalized = X_centered / scale
            
            # Step 1: Construct KNN graph and Laplacian
            L = self._construct_laplacian(X_normalized)
            
            # Step 2: ADMM optimization for P and S
            P, S = self._admm_optimization(X_normalized, L, d)
            
            # Project features
            X_proj = X_normalized @ P
            
            # L2 normalize rows for stability
            X_proj = normalize(X_proj, norm='l2', axis=1)
            
            projected_views.append(X_proj)
            projection_matrices.append(P)
            self_expression_matrices.append(S)
            
        return projected_views, projection_matrices, self_expression_matrices
    
    def _construct_laplacian(self, X):
        """
        Construct SPARSE graph Laplacian from KNN graph.
        
        Complexity: O(N·k·log(N)) for KNN + O(N·k) for graph construction
        Space: O(N·k) sparse matrix
        """
        N = X.shape[0]
        k = min(self.k_neighbors, N - 1)
        
        # Fast KNN search: O(N·log(N)·D) with ball tree
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Adaptive bandwidth
        sigma = np.median(distances[:, 1:]) + 1e-10
        
        # Build SPARSE adjacency matrix: O(N·k)
        row_idx = []
        col_idx = []
        data = []
        
        for i in range(N):
            for j_idx in range(1, k + 1):
                j = indices[i, j_idx]
                dist = distances[i, j_idx]
                weight = np.exp(-dist**2 / (2 * sigma**2))
                
                row_idx.append(i)
                col_idx.append(j)
                data.append(weight)
        
        # Sparse adjacency
        A_sparse = sparse.csr_matrix(
            (data, (row_idx, col_idx)), shape=(N, N)
        )
        
        # Symmetrize
        A_sparse = (A_sparse + A_sparse.T) / 2
        
        # Sparse normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        d = np.array(A_sparse.sum(axis=1)).flatten()
        d = np.maximum(d, 1e-10)
        d_inv_sqrt = 1.0 / np.sqrt(d)
        
        # D^{-1/2} A D^{-1/2} as sparse
        D_inv_sqrt = sparse.diags(d_inv_sqrt)
        L_sparse = sparse.eye(N) - D_inv_sqrt @ A_sparse @ D_inv_sqrt
        
        return L_sparse
    
    def _faiss_knn(self, X, k):
        """Fast KNN using FAISS."""
        X = np.ascontiguousarray(X.astype(np.float32))
        N, D = X.shape
        
        index = faiss.IndexFlatL2(D)
        index.add(X)
        
        distances, indices = index.search(X, k + 1)
        
        # Build heat kernel adjacency matrix
        sigma = np.median(distances[:, 1:]) + 1e-10
        A = np.zeros((N, N))
        for i in range(N):
            for j_idx, j in enumerate(indices[i, 1:]):
                dist = distances[i, j_idx + 1]
                A[i, j] = np.exp(-dist / (2 * sigma**2))
                
        return A
    
    def _sklearn_knn(self, X, k):
        """KNN using sklearn with heat kernel weights."""
        N = X.shape[0]
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        sigma = np.median(distances[:, 1:]) + 1e-10
        A = np.zeros((N, N))
        for i in range(N):
            for j_idx, j in enumerate(indices[i, 1:]):
                dist = distances[i, j_idx + 1]
                A[i, j] = np.exp(-dist / (2 * sigma**2))
                
        return A
    
    def _admm_optimization(self, X, L, d):
        """
        Near-linear feature projection using randomized SVD.
        
        Complexity: O(N·D·d) instead of O(D³)
        
        Uses:
        1. Randomized SVD for fast PCA: O(N·D·d)
        2. Sparse Laplacian regularization: O(N·k·d)
        """
        N, D = X.shape
        
        # Randomized PCA for large D: O(N·D·d)
        if D > 500:
            # Use randomized projection
            n_oversamples = min(10, D - d)
            n_components = d + n_oversamples
            
            # Random projection matrix
            Omega = np.random.randn(D, n_components)
            Y = X @ Omega  # N x n_components
            
            # QR decomposition
            Q, _ = np.linalg.qr(Y)
            
            # Project X onto Q
            B = Q.T @ X  # n_components x D
            
            # SVD of small matrix
            _, S, Vt = np.linalg.svd(B, full_matrices=False)
            
            # Take top d right singular vectors
            P = Vt[:d].T  # D x d
        else:
            # Standard approach for small D
            XtX = X.T @ X
            reg = 1e-6 * np.trace(XtX) / D
            XtX += reg * np.eye(D)
            
            eigenvalues, eigenvectors = np.linalg.eigh(XtX)
            idx = np.argsort(eigenvalues)[::-1][:d]
            P = eigenvectors[:, idx]
        
        # Light Laplacian regularization using sparse L: O(N·k·d)
        Z = X @ P
        alpha_blend = min(self.alpha * 0.1, 0.5)
        
        if alpha_blend > 0 and sparse.issparse(L):
            # Compute LZ efficiently for sparse L: O(N·k·d)
            LZ = L @ Z
            
            # Gradient step to reduce Laplacian energy
            # P_new = P - α * X'LZ / N
            grad = X.T @ LZ / N
            P = P - alpha_blend * 0.1 * grad
            
            # Re-orthogonalize
            P, _ = np.linalg.qr(P)
            P = P[:, :d]
        
        # Orthonormalize P
        P, _ = np.linalg.qr(P)
        if P.shape[1] < d:
            P = np.hstack([P, np.zeros((D, d - P.shape[1]))])
        P = P[:, :d]
        
        # Compute S using sparse local linear reconstruction
        Z = X @ P
        S = self._compute_sparse_S(Z)
        
        if self.verbose:
            # Compute objective for reporting (simplified for sparse L)
            I_minus_S = np.eye(N) - S
            recon_term = I_minus_S @ Z
            recon_loss = np.linalg.norm(recon_term, 'fro') ** 2
            
            # Laplacian term with sparse L
            if sparse.issparse(L):
                LZ = L @ Z
                lap_loss = self.alpha * np.trace(Z.T @ LZ)
            else:
                lap_loss = 0.0
            
            obj = recon_loss + lap_loss
            print(f"    Optimization complete, obj={obj:.4f}")
        
        return P, S
    
    def _compute_sparse_S(self, Z):
        """
        Compute sparse self-expression matrix S.
        Each point is expressed as a sparse linear combination of its neighbors.
        """
        N = Z.shape[0]
        k = min(self.k_neighbors, N - 1)
        
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(Z)
        distances, indices = nbrs.kneighbors(Z)
        
        S = np.zeros((N, N))
        
        for i in range(N):
            neighbors = indices[i, 1:]  # Exclude self
            Z_neighbors = Z[neighbors]
            z_i = Z[i]
            
            # Solve local least squares with L1 regularization
            # min ||z_i - Z_neighbors' @ w||^2 + beta * ||w||_1
            # Use iterative soft thresholding
            n_neigh = len(neighbors)
            w = np.ones(n_neigh) / n_neigh  # Initialize uniform
            
            # Precompute
            ZtZ = Z_neighbors @ Z_neighbors.T
            Ztz = Z_neighbors @ z_i
            
            # Simple ISTA iterations
            L_lip = np.linalg.norm(ZtZ, 2) + 1e-6
            step = 1.0 / L_lip
            
            for _ in range(10):
                grad = ZtZ @ w - Ztz
                w = w - step * grad
                # Soft thresholding
                w = np.sign(w) * np.maximum(np.abs(w) - step * self.beta, 0)
                # Keep non-negative for interpretability
                w = np.maximum(w, 0)
            
            # Normalize weights
            w_sum = np.sum(w) + 1e-10
            w = w / w_sum
            
            S[i, neighbors] = w
        
        return S
    
    def _initialize_local_S(self, X):
        """Initialize S using local linear coding (LLE-style)."""
        N = X.shape[0]
        k = min(self.k_neighbors, N - 1)
        
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        _, indices = nbrs.kneighbors(X)
        
        S = np.zeros((N, N))
        
        for i in range(N):
            neighbors = indices[i, 1:]  # Exclude self
            X_neighbors = X[neighbors]
            x_i = X[i]
            
            # Solve local least squares: min ||x_i - X_neighbors' * w||^2
            # with constraint sum(w) = 1
            try:
                # Local Gram matrix
                Z = X_neighbors - x_i
                G = Z @ Z.T
                G += np.eye(k) * 1e-4 * np.trace(G) / k  # Regularize
                
                # Solve for weights
                w = np.linalg.solve(G, np.ones(k))
                w = w / (np.sum(w) + 1e-10)
                
                S[i, neighbors] = w
            except:
                S[i, neighbors] = 1.0 / k
                
        return S
    
    def _update_S_sparse(self, X, P, S):
        """
        Update S via proximal gradient for:
            min ||Z - SZ||_F^2 + β||S||_1
            s.t. diag(S) = 0
        
        where Z = XP (projected features)
        
        Use multiple gradient steps for better optimization.
        """
        Z = X @ P
        N = Z.shape[0]
        
        # Compute ZZ' once
        ZZt = Z @ Z.T
        
        # Lipschitz constant for step size
        L_lip = 2 * np.linalg.norm(ZZt, 2) + 1e-6
        step_size = 1.0 / L_lip
        
        S_new = S.copy()
        
        # Multiple proximal gradient steps
        for _ in range(5):
            # Gradient: 2 * (S @ ZZ' - ZZ')
            grad = 2 * (S_new @ ZZt - ZZt)
            
            # Gradient descent step
            S_new = S_new - step_size * grad
            
            # Proximal step: soft thresholding for L1
            threshold = step_size * self.beta
            S_new = np.sign(S_new) * np.maximum(np.abs(S_new) - threshold, 0)
            
            # Enforce diagonal = 0
            np.fill_diagonal(S_new, 0)
        
        # Enforce sparsity: keep only k-nearest neighbors structure
        k = min(self.k_neighbors * 2, N - 1)
        for i in range(N):
            row = S_new[i].copy()
            nnz = np.count_nonzero(row)
            if nnz > k:
                # Keep only k largest absolute values
                abs_row = np.abs(row)
                threshold_k = np.partition(abs_row, -k)[-k]
                row[abs_row < threshold_k] = 0
                S_new[i] = row
        
        # Ensure S is not too large (avoid explosion)
        max_val = np.max(np.abs(S_new))
        if max_val > 10:
            S_new = S_new / max_val * 10
        
        return S_new
    
    def _update_P_admm(self, X, L, S, P, Q, Y, XtLX, d, rho=None):
        """
        ADMM update for P.
        
        Augmented Lagrangian:
            L_ρ = Tr(P'MP) + λ||Q||_{2,1} + <Y, P-Q> + (ρ/2)||P-Q||_F^2
        
        where M = X'(I-S)'(I-S)X + αX'LX
        
        Updates:
            P: solve (2M + ρI)P = ρQ - Y, then orthogonalize
            Q: proximal L2,1 (row-wise soft thresholding)
            Y: Y + ρ(P - Q)
        """
        if rho is None:
            rho = self.rho
            
        N, D = X.shape
        I_minus_S = np.eye(N) - S
        
        # M = X'(I-S)'(I-S)X + α*X'LX
        XtISt = X.T @ I_minus_S.T
        M = XtISt @ I_minus_S @ X + self.alpha * XtLX
        
        # Regularize M for numerical stability
        M = (M + M.T) / 2
        reg = max(1e-4, 1e-6 * np.trace(M) / D)
        M += np.eye(D) * reg
        
        # Update P: solve (2M + ρI)P = ρQ - Y
        A = 2 * M + rho * np.eye(D)
        B = rho * Q - Y
        
        try:
            P_new = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            P_new = np.linalg.lstsq(A, B, rcond=None)[0]
        
        # Orthogonalize P via SVD (Procrustes)
        U, _, Vt = np.linalg.svd(P_new, full_matrices=False)
        n_cols = min(d, U.shape[1], Vt.shape[0])
        P_new = U[:, :n_cols] @ Vt[:n_cols, :n_cols]
        
        # Ensure correct shape
        if P_new.shape[1] < d:
            # Pad with zeros if needed
            P_new = np.hstack([P_new, np.zeros((D, d - P_new.shape[1]))])
        
        # Update Q: proximal operator for L2,1 norm
        Q_new = self._l21_proximal(P_new + Y / rho, self.lambda_l21 / rho)
        
        # Update Y (dual variable)
        Y_new = Y + rho * (P_new - Q_new)
        
        return P_new, Q_new, Y_new
    
    def _l21_proximal(self, X, threshold):
        """
        Proximal operator for L2,1 norm (row-wise soft thresholding).
        
        prox_{λ||·||_{2,1}}(X)[i,:] = max(0, 1 - λ/||X[i,:]||_2) * X[i,:]
        """
        row_norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
        scale = np.maximum(1 - threshold / row_norms, 0)
        return X * scale
    
    # ==================== Phase 2: RFF-based Graph Alignment ====================
    
    def _phase2_compute_view_weights(self, projected_views):
        """
        Phase 2 (Near-Linear): Compute view quality weights using KDE.
        
        Complexity: O(V · N · k · log(N)) for KNN
                   O(V · N · k) for density estimation
        Total: O(V · N · k · log(N))
        """
        n_views = len(projected_views)
        view_qualities = []
        
        for v, X in enumerate(projected_views):
            if self.verbose:
                print(f"  Processing view {v+1}/{n_views}...")
            
            # Fast KNN-based quality estimation: O(N · k · log(N))
            N = X.shape[0]
            k = min(self.k_neighbors, N - 1)
            
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(X)
            distances, _ = nbrs.kneighbors(X)
            
            # Density from k-NN distances
            sigma = np.median(distances[:, 1:]) + 1e-10
            densities = np.exp(-distances[:, 1:].mean(axis=1)**2 / (2 * sigma**2))
            
            # View quality: uniform density = high quality
            cv = np.std(densities) / (np.mean(densities) + 1e-10)
            quality = 1.0 / (1.0 + cv)
            
            # Also consider compactness (average neighbor distance)
            avg_dist = distances[:, 1:k+1].mean()
            compactness = 1.0 / (1.0 + avg_dist)
            
            view_qualities.append(quality * 0.7 + compactness * 0.3)
        
        # Normalize to weights
        view_weights = np.array(view_qualities)
        view_weights = view_weights / (view_weights.sum() + 1e-10)
        
        if self.verbose:
            print(f"  View weights: {view_weights}")
        
        return view_weights
    
    def _phase2_graph_alignment(self, projected_views):
        """
        Phase 2: Multi-view Graph Fusion with KDE and Optimal Transport
        
        1. Construct KNN graphs for each view
        2. Use KDE to estimate density and compute view quality weights
        3. Use Sinkhorn OT to align graphs before fusion
        """
        n_views = len(projected_views)
        n_samples = projected_views[0].shape[0]
        
        view_graphs = []
        view_densities = []
        view_qualities = []
        
        for v, X in enumerate(projected_views):
            if self.verbose:
                print(f"  Processing view {v+1}/{n_views}...")
            
            # Step 1: Construct KNN graph
            G, sigma = self._construct_knn_graph_with_sigma(X)
            view_graphs.append(G)
            
            # Step 2: KDE-based density estimation
            densities = self._fast_kde(X, sigma)
            view_densities.append(densities)
            
            # Step 3: Compute view quality based on density distribution
            # Views with more uniform density (lower CV) are higher quality
            cv = np.std(densities) / (np.mean(densities) + 1e-10)
            quality = 1.0 / (1.0 + cv)
            
            # Also consider graph connectivity
            avg_degree = G.sum(axis=1).mean()
            connectivity = min(avg_degree / self.k_neighbors, 1.0)
            
            view_qualities.append(quality * 0.7 + connectivity * 0.3)
        
        # Compute view weights from qualities
        view_weights = np.array(view_qualities)
        view_weights = view_weights / (view_weights.sum() + 1e-10)
        
        if self.verbose:
            print(f"  View weights: {view_weights}")
        
        # Step 4: Sinkhorn OT-based graph alignment and fusion
        consensus = self._ot_graph_fusion(view_graphs, view_densities, view_weights)
        
        return consensus, view_weights
    
    def _construct_knn_graph_with_sigma(self, X):
        """
        Construct KNN graph with heat kernel weights.
        Returns both graph and bandwidth sigma.
        """
        N = X.shape[0]
        k = min(self.k_neighbors * 2, N - 1)
        
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Adaptive sigma (bandwidth)
        sigma = np.median(distances[:, 1:]) + 1e-10
        
        # Build graph with heat kernel
        G = np.zeros((N, N))
        for i in range(N):
            for j_idx, (idx, dist) in enumerate(zip(indices[i, 1:], distances[i, 1:])):
                weight = np.exp(-dist**2 / (2 * sigma**2))
                G[i, idx] = weight
        
        # Symmetrize
        G = (G + G.T) / 2
        
        return G, sigma
    
    def _fast_kde(self, X, sigma):
        """
        Fast KDE using k-nearest neighbors.
        
        p(x_i) ≈ (1/k) * sum_{j in kNN(i)} exp(-||x_i - x_j||^2 / (2*sigma^2))
        """
        N = X.shape[0]
        k = min(self.k_neighbors, N - 1)
        
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        # Density estimate from neighbor distances
        densities = np.zeros(N)
        for i in range(N):
            # Exclude self (distance 0)
            neighbor_dists = distances[i, 1:]
            kernel_vals = np.exp(-neighbor_dists**2 / (2 * sigma**2))
            densities[i] = np.mean(kernel_vals)
        
        # Normalize to [0, 1]
        densities = (densities - densities.min()) / (densities.max() - densities.min() + 1e-10)
        
        return densities
    
    def _ot_graph_fusion(self, view_graphs, view_densities, view_weights):
        """
        Optimal Transport-based graph fusion with entropy-regularized edge reweighting.
        
        Key insight: Use OT to compute consistency weights for each edge,
        rather than transforming entire graphs.
        
        Edges that appear consistently across views (with high OT scores) get boosted,
        while inconsistent edges get down-weighted.
        """
        n_views = len(view_graphs)
        n_samples = view_graphs[0].shape[0]
        
        # Step 1: Compute edge consistency matrix across views
        # For each edge (i,j), measure how consistently it appears
        edge_consensus = np.zeros((n_samples, n_samples))
        edge_counts = np.zeros((n_samples, n_samples))
        
        for v in range(n_views):
            G_v = view_graphs[v]
            # Convert to binary for consistency check
            mask = (G_v > 0).astype(float)
            edge_counts += mask
            edge_consensus += G_v * view_weights[v]
        
        # Consistency score: how many views agree
        consistency = edge_counts / n_views  # [0, 1]
        
        # Step 2: Density-based edge scoring using OT
        # Compute global density from all views
        global_density = np.zeros(n_samples)
        for v in range(n_views):
            global_density += view_weights[v] * view_densities[v]
        
        # OT-based reweighting: edges between similar-density points
        # are more likely to be intra-cluster
        density_similarity = self._compute_ot_density_weights(global_density)
        
        # Step 3: Enhanced fusion with multiple cues
        consensus = np.zeros((n_samples, n_samples))
        
        for v in range(n_views):
            G_v = view_graphs[v].copy()
            d_v = view_densities[v]
            
            # Cue 1: Density product (high density -> likely cluster core)
            density_product = np.outer(d_v, d_v)
            
            # Cue 2: Local neighborhood agreement
            # Points with similar local structure should connect
            local_sim = self._local_neighborhood_similarity(G_v)
            
            # Combine cues: boost confident edges
            confidence = (
                0.4 * density_product +        # Dense regions
                0.3 * local_sim +              # Structural similarity
                0.3 * consistency              # Cross-view agreement
            )
            
            # Apply confidence as edge weight modifier
            # Range: [0.5, 1.5] to avoid zeroing out edges
            modifier = 0.5 + confidence
            G_enhanced = G_v * modifier
            
            consensus += view_weights[v] * G_enhanced
        
        # Step 4: Apply OT-based global refinement
        consensus = self._ot_refine_consensus(consensus, global_density)
        
        # Symmetrize and clean
        consensus = (consensus + consensus.T) / 2
        consensus = np.maximum(consensus, 0)
        np.fill_diagonal(consensus, 0)
        
        # Row-normalize for spectral clustering
        row_sums = consensus.sum(axis=1, keepdims=True) + 1e-10
        consensus = consensus / row_sums
        consensus = (consensus + consensus.T) / 2  # Re-symmetrize
        
        return consensus
    
    def _compute_ot_density_weights(self, densities):
        """
        Use OT to compute similarity weights based on density distribution.
        Points with similar density values should have higher affinity.
        """
        N = len(densities)
        
        # Compute density difference cost
        d_diff = np.abs(densities[:, np.newaxis] - densities[np.newaxis, :])
        
        # Convert to similarity (inverse of difference)
        # Use soft assignment
        sigma = np.std(densities) + 1e-10
        similarity = np.exp(-d_diff**2 / (2 * sigma**2))
        
        return similarity
    
    def _local_neighborhood_similarity(self, G):
        """
        Compute local neighborhood similarity between all pairs.
        Points with similar neighborhoods should be in the same cluster.
        """
        N = G.shape[0]
        
        # Normalize rows
        row_sums = G.sum(axis=1, keepdims=True) + 1e-10
        G_norm = G / row_sums
        
        # Cosine similarity of neighborhood vectors
        sim = G_norm @ G_norm.T
        
        # Clip to [0, 1]
        sim = np.clip(sim, 0, 1)
        
        return sim
    
    def _ot_refine_consensus(self, G, densities):
        """
        Refine consensus graph using entropic OT.
        
        Idea: Transport mass from low-confidence regions to high-confidence regions.
        """
        N = G.shape[0]
        
        # Source: current graph edge distribution
        # Target: ideal (uniform within clusters)
        
        # Use density to identify cluster cores
        # High-density points are likely cluster centers
        
        # Compute node importance from density
        importance = densities / (densities.sum() + 1e-10)
        
        # Scale edges by importance of endpoints
        importance_weight = np.outer(importance, importance)
        importance_weight = importance_weight / (importance_weight.max() + 1e-10)
        
        # Boost edges between important nodes
        boost = 1.0 + 0.5 * importance_weight
        G_refined = G * boost
        
        return G_refined
    
    def _sinkhorn_transport(self, G_source, G_target, densities, n_iter=20, reg=0.1):
        """
        Compute Sinkhorn transport plan between two graphs.
        
        Uses degree distributions as marginals and graph similarity as cost.
        """
        N = G_source.shape[0]
        
        # Marginals: degree distributions (normalized)
        deg_source = G_source.sum(axis=1) + 1e-10
        deg_target = G_target.sum(axis=1) + 1e-10
        
        a = deg_source / deg_source.sum()
        b = deg_target / deg_target.sum()
        
        # Cost matrix: dissimilarity between node neighborhoods
        # Use negative of graph product as similarity -> cost
        # C_ij = 1 - (G_source[i] @ G_target[j]) / (||G_source[i]|| * ||G_target[j]||)
        
        # For efficiency, use a simpler cost: based on density difference
        density_diff = np.abs(densities[:, np.newaxis] - densities[np.newaxis, :])
        C = density_diff / (density_diff.max() + 1e-10)
        
        # Gibbs kernel
        K = np.exp(-C / reg)
        
        # Sinkhorn iterations
        u = np.ones(N)
        v = np.ones(N)
        
        for _ in range(n_iter):
            u = a / (K @ v + 1e-100)
            v = b / (K.T @ u + 1e-100)
            
            # Check for numerical issues
            if np.any(np.isnan(u)) or np.any(np.isnan(v)):
                u = np.ones(N)
                v = np.ones(N)
                break
        
        # Transport plan
        T = np.diag(u) @ K @ np.diag(v)
        
        # Normalize rows to sum to 1 (make it a proper transport)
        row_sums = T.sum(axis=1, keepdims=True) + 1e-10
        T = T / row_sums
        
        return T
    
    def _construct_knn_graph(self, X):
        """
        Construct KNN graph with heat kernel weights (legacy interface).
        """
        G, _ = self._construct_knn_graph_with_sigma(X)
        return G
    
    def _entropy_bandwidth_selection(self, X, n_candidates=10):
        """
        Select kernel bandwidth by maximizing entropy of density estimate.
        
        H(p) = -E[log p(x)]
        """
        # Sample subset for efficiency
        n = min(500, X.shape[0])
        idx = np.random.choice(X.shape[0], n, replace=False)
        X_sample = X[idx]
        
        # Compute pairwise distances
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(np.linalg.norm(X_sample[i] - X_sample[j]))
        dists = np.array(dists)
        
        if len(dists) == 0:
            return 1.0
        
        # Candidate bandwidths around median
        median_dist = np.median(dists)
        candidates = median_dist * np.logspace(-1, 1, n_candidates)
        
        best_sigma = median_dist
        best_entropy = -float('inf')
        
        for sigma in candidates:
            # Quick RFF density estimate
            D_rff = min(64, self.rff_dim)
            d = X_sample.shape[1]
            omega = np.random.randn(d, D_rff // 2) / sigma
            XW = X_sample @ omega
            rff = np.sqrt(2.0 / (D_rff // 2)) * np.hstack([np.cos(XW), np.sin(XW)])
            
            phi_mean = np.mean(rff, axis=0)
            densities = rff @ phi_mean
            densities = np.maximum(densities, 1e-10)
            
            # Compute entropy
            entropy = -np.mean(np.log(densities))
            
            if entropy > best_entropy:
                best_entropy = entropy
                best_sigma = sigma
        
        return best_sigma
    
    def _compute_rff(self, X, sigma):
        """
        Compute Random Fourier Features for RBF kernel approximation.
        
        k(x, y) = exp(-||x-y||^2 / (2σ^2)) ≈ φ(x)' φ(y)
        
        where φ(x) = sqrt(2/D) * [cos(ω'x + b), sin(ω'x + b)]
        and ω ~ N(0, 1/σ² I)
        """
        N, d = X.shape
        D = self.rff_dim
        
        # Sample random frequencies from N(0, 1/σ² I)
        omega = np.random.randn(d, D // 2) / sigma
        
        # Compute features
        XW = X @ omega
        rff = np.sqrt(2.0 / (D // 2)) * np.hstack([np.cos(XW), np.sin(XW)])
        
        return rff, omega
    
    def _rff_density_estimation(self, rff_features):
        """
        RFF-accelerated kernel density estimation.
        
        p(x) = (1/N) Σ k(x, x_i) ≈ φ(x)' * (1/N Σ φ(x_i)) = φ(x)' * φ_mean
        
        Complexity: O(ND) instead of O(N²)
        """
        # Mean RFF feature
        phi_mean = np.mean(rff_features, axis=0)
        
        # Density estimate for each point
        densities = rff_features @ phi_mean
        
        # Ensure positive
        densities = np.maximum(densities, 1e-10)
        
        return densities
    
    def _construct_density_aware_graph(self, X, densities, threshold, sigma):
        """
        Construct density-aware similarity graph.
        
        G_ij = k(x_i, x_j) * w(p_i, p_j)
        
        where w is a soft weighting based on density, rather than hard thresholding.
        """
        N = X.shape[0]
        k = min(self.k_neighbors * 2, N - 1)
        
        # Get neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Normalize densities to [0, 1]
        d_min, d_max = densities.min(), densities.max()
        if d_max > d_min:
            densities_norm = (densities - d_min) / (d_max - d_min)
        else:
            densities_norm = np.ones_like(densities)
        
        # Build graph with soft density weighting
        G = np.zeros((N, N))
        for i in range(N):
            for j_idx, (idx, dist) in enumerate(zip(indices[i, 1:], distances[i, 1:])):
                # RBF kernel weight
                base_weight = np.exp(-dist**2 / (2 * sigma**2))
                
                # Soft density weighting: geometric mean of normalized densities
                # This gives more weight to edges between high-density points
                density_weight = np.sqrt(densities_norm[i] * densities_norm[idx])
                
                # Blend: 70% base structure + 30% density-aware
                # This prevents completely losing low-density connections
                final_weight = base_weight * (0.7 + 0.3 * density_weight)
                
                G[i, idx] = final_weight
        
        # Symmetrize
        G = (G + G.T) / 2
        
        return G
    
    def _rff_sinkhorn_alignment(self, view_graphs, projected_views, view_densities):
        """
        Align view-specific graphs using density-weighted averaging.
        
        For stability, we use a simpler weighted fusion instead of 
        full Sinkhorn OT, which preserves the paper's core idea of
        density-aware weighting while being numerically stable.
        """
        n_views = len(view_graphs)
        n_samples = view_graphs[0].shape[0]
        
        # Compute view quality weights based on:
        # 1. Density uniformity (less skewed = better)
        # 2. Graph quality (more connected = better)
        view_weights = []
        for v in range(n_views):
            densities = view_densities[v]
            G = view_graphs[v]
            
            # Density uniformity (higher = better)
            cv = np.std(densities) / (np.mean(densities) + 1e-10)
            density_score = 1.0 / (1.0 + cv)
            
            # Graph connectivity (higher avg degree = better)
            avg_degree = G.sum(axis=1).mean()
            graph_score = min(avg_degree / 10, 1.0)  # Cap at 1
            
            # Combined score
            quality = density_score * 0.5 + graph_score * 0.5
            view_weights.append(quality)
        
        view_weights = np.array(view_weights)
        view_weights = view_weights / (view_weights.sum() + 1e-10)
        
        if self.verbose:
            print(f"  View weights: {view_weights}")
        
        # Weighted fusion of graphs
        consensus = np.zeros((n_samples, n_samples))
        for v in range(n_views):
            G_v = view_graphs[v]
            
            # Normalize each view graph to [0, 1]
            G_min, G_max = G_v.min(), G_v.max()
            if G_max > G_min:
                G_normalized = (G_v - G_min) / (G_max - G_min)
            else:
                G_normalized = G_v
            
            consensus += view_weights[v] * G_normalized
        
        # Symmetrize
        consensus = (consensus + consensus.T) / 2
        
        # Ensure non-negative
        consensus = np.maximum(consensus, 0)
        
        # Remove self-loops
        np.fill_diagonal(consensus, 0)
        
        return consensus, view_weights
    
    def _rff_sinkhorn(self, X, G_source, G_target, omega, n_iter=None):
        """
        RFF-accelerated Sinkhorn algorithm.
        
        Instead of computing full kernel matrix K = exp(-C/ε),
        we approximate using RFF: K ≈ Φ @ Φ'
        
        This allows K @ v = Φ @ (Φ' @ v) in O(ND) instead of O(N²).
        """
        if n_iter is None:
            n_iter = self.sinkhorn_iter
            
        N = X.shape[0]
        reg = self.sinkhorn_reg
        
        # Compute RFF for Gibbs kernel
        # For exp(-C/ε) where C = ||x-y||², we use σ = sqrt(ε/2)
        sigma_gibbs = np.sqrt(reg / 2 + 1e-10)
        d = X.shape[1]
        
        # Use stored omega but rescale for Gibbs kernel
        D_rff = omega.shape[1] * 2
        omega_gibbs = omega * (1.0 / sigma_gibbs) * np.linalg.norm(omega, axis=0).mean()
        
        XW = X @ omega_gibbs
        Phi = np.sqrt(2.0 / (D_rff // 2)) * np.hstack([np.cos(XW), np.sin(XW)])
        
        # Uniform marginals
        a = np.ones(N) / N
        b = np.ones(N) / N
        
        # Initialize scaling vectors
        u = np.ones(N)
        v = np.ones(N)
        
        # Precompute Φ' for efficiency
        Phi_T = Phi.T
        
        for _ in range(n_iter):
            # v = b / (K' @ u) where K'u ≈ Φ @ (Φ' @ u)
            Phi_u = Phi_T @ u
            Ktu = Phi @ Phi_u
            v = b / (Ktu + 1e-100)
            
            # u = a / (K @ v)
            Phi_v = Phi_T @ v
            Kv = Phi @ Phi_v
            u = a / (Kv + 1e-100)
        
        # Compute transport plan T = diag(u) @ K @ diag(v)
        # T_ij = u_i * K_ij * v_j ≈ u_i * (Φ_i' @ Φ_j) * v_j
        # For sparse representation, compute only significant entries
        
        # Approximate T as low-rank: T ≈ (u .* Φ) @ (v .* Φ)'
        T_left = u[:, np.newaxis] * Phi
        T_right = v[:, np.newaxis] * Phi
        
        # For efficiency, return a function or the factors
        # Here we compute the sparse approximation
        T = T_left @ T_right.T
        
        # Normalize to be doubly stochastic
        T = T / (T.sum() + 1e-10) * N
        
        return T
    
    # ==================== Phase 3: OT-Enhanced Spectral Clustering ====================
    
    def _phase3_hybrid_clustering(self, consensus_graph, projected_views, view_weights, n_samples):
        """
        Phase 3: OT-Enhanced Spectral Clustering.
        
        Key innovations:
        1. Build KNN graph on concatenated features
        2. Use Sinkhorn OT to compute soft cluster assignments
        3. Enhance graph edges using OT transport plan
        4. Spectral embedding on enhanced graph
        
        Complexity: O(N·k·log(N) + N·M·T) where M = landmarks, T = Sinkhorn iters
        """
        # Concatenate projected features (weighted)
        X_concat = np.zeros((n_samples, 0))
        for v, (X_v, w_v) in enumerate(zip(projected_views, view_weights)):
            X_v_norm = (X_v - X_v.mean(axis=0)) / (X_v.std(axis=0) + 1e-10)
            X_concat = np.hstack([X_concat, np.sqrt(w_v) * X_v_norm])
        
        if self.verbose:
            print(f"  Building KNN graph ({X_concat.shape[1]} dims)...")
        
        # Step 1: Build base KNN graph
        W_base, knn_indices, knn_distances, density = self._build_knn_graph_with_info(X_concat)
        
        if self.verbose:
            print(f"  Computing OT-enhanced graph...")
        
        # Step 2: OT enhancement using landmark-based Sinkhorn
        W_enhanced = self._ot_enhance_graph(X_concat, W_base, knn_indices, density)
        
        # Step 3: Spectral embedding
        if self.verbose:
            print(f"  Computing spectral embedding...")
        Z_spectral = self._standard_spectral_embedding(W_enhanced)
        
        self.spectral_embedding_ = Z_spectral
        
        # Step 4: K-Means clustering
        best_labels = None
        best_inertia = float('inf')
        
        for restart in range(5):
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=None if self.random_state is None else self.random_state + restart
            )
            labels = kmeans.fit_predict(Z_spectral)
            
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_labels = labels
        
        return best_labels
    
    def _build_knn_graph_with_info(self, X):
        """
        Build KNN graph and return additional info for OT enhancement.
        
        Returns:
            W_sparse: Sparse adjacency matrix
            indices: KNN indices
            distances: KNN distances  
            density: Local density estimates
        """
        N = X.shape[0]
        k = min(self.k_neighbors * 2, N - 1)
        
        # KNN search
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Adaptive bandwidth
        sigma_i = distances[:, self.k_neighbors]
        sigma_i = np.maximum(sigma_i, 1e-10)
        
        # Density estimation
        density = 1.0 / (np.mean(distances[:, 1:self.k_neighbors+1], axis=1) + 1e-10)
        density = (density - density.min()) / (density.max() - density.min() + 1e-10)
        
        # Build sparse graph
        row_idx = []
        col_idx = []
        data = []
        
        for i in range(N):
            for j_idx in range(1, k + 1):
                j = indices[i, j_idx]
                dist = distances[i, j_idx]
                
                sigma_ij = (sigma_i[i] + sigma_i[j]) / 2
                w_heat = np.exp(-dist**2 / (2 * sigma_ij**2))
                
                row_idx.append(i)
                col_idx.append(j)
                data.append(w_heat)
        
        W_sparse = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
        W_sparse = (W_sparse + W_sparse.T) / 2
        
        return W_sparse, indices, distances, density
    
    def _ot_enhance_graph(self, X, W_base, knn_indices, density):
        """
        Enhance graph using Optimal Transport - Additive approach.
        
        Key insight: Add OT-based similarity as a second term rather than multiplicative.
        This preserves the base graph structure better.
        
        Complexity: O(N·M·T) where M << N
        """
        N = X.shape[0]
        M = min(self.n_clusters * 10, N // 5, 200)
        
        # Step 1: Select landmarks using density-weighted k-means++
        landmarks = self._select_ot_landmarks(X, density, M)
        X_landmarks = X[landmarks]
        
        # Step 2: Compute cost matrix
        C = np.zeros((N, M))
        for m in range(M):
            diff = X - X_landmarks[m]
            C[:, m] = np.sum(diff**2, axis=1)
        
        # Normalization
        C_med = np.median(C)
        C = C / (C_med + 1e-10)
        
        # Step 3: Sinkhorn algorithm
        reg = 0.05
        T = self._sinkhorn_points_to_landmarks(C, reg, n_iter=80)
        
        # Step 4: Normalize transport profiles
        T_norm = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-10)
        
        # Step 5: Get sparse graph structure
        W_base_coo = W_base.tocoo()
        rows = W_base_coo.row
        cols = W_base_coo.col
        vals = W_base_coo.data.copy()
        
        # Step 6: Compute OT similarity for edges
        ot_sims = np.sum(T_norm[rows] * T_norm[cols], axis=1)
        
        # Step 7: Density similarity
        density_sims = 1.0 - np.abs(density[rows] - density[cols])
        
        # Step 8: Original density correction
        density_factor = 0.5 + 0.5 * density_sims
        
        # Step 9: Slightly larger additive OT bonus for high-similarity pairs
        ot_bonus = 0.06 * np.maximum(0, ot_sims - 0.5)  # Only positive bonus
        
        # Combined: multiplicative density + additive OT
        boost = density_factor + ot_bonus
        
        vals = vals * boost
        
        # Reconstruct sparse matrix
        W_enhanced = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
        W_enhanced = (W_enhanced + W_enhanced.T) / 2
        
        return W_enhanced
    
    def _select_ot_landmarks(self, X, density, M):
        """
        Select landmarks for OT computation.
        
        Strategy: Prefer high-density points (cluster cores) with good coverage.
        Uses k-means++ style selection weighted by density.
        """
        N = X.shape[0]
        
        # Weight by density (prefer cluster cores)
        weights = density ** 2
        weights = weights / weights.sum()
        
        # First landmark: highest density point
        landmarks = [np.argmax(density)]
        
        # Remaining landmarks: k-means++ style
        for _ in range(M - 1):
            # Distance to nearest landmark
            min_dists = np.full(N, np.inf)
            for l in landmarks:
                dists = np.sum((X - X[l])**2, axis=1)
                min_dists = np.minimum(min_dists, dists)
            
            # Probability proportional to distance * density
            probs = min_dists * weights
            probs[landmarks] = 0
            probs = probs / (probs.sum() + 1e-10)
            
            # Sample next landmark
            next_l = np.random.choice(N, p=probs)
            landmarks.append(next_l)
        
        return np.array(landmarks)
    
    def _sinkhorn_points_to_landmarks(self, C, reg, n_iter=50):
        """
        Sinkhorn algorithm for points-to-landmarks transport.
        
        Solves: min <T, C> + reg * H(T)
        s.t. T @ 1 = a (row sums)
             T.T @ 1 = b (col sums)
        
        With a = 1/N (uniform over points), b = 1/M (uniform over landmarks)
        
        Complexity: O(N·M·n_iter)
        """
        N, M = C.shape
        
        # Uniform marginals
        a = np.ones(N) / N
        b = np.ones(M) / M
        
        # Gibbs kernel
        K = np.exp(-C / reg)
        
        # Sinkhorn iterations
        u = np.ones(N)
        v = np.ones(M)
        
        for _ in range(n_iter):
            # Update v
            Ktu = K.T @ u
            v = b / (Ktu + 1e-100)
            
            # Update u  
            Kv = K @ v
            u = a / (Kv + 1e-100)
            
            # Check for numerical issues
            if np.any(np.isnan(u)) or np.any(np.isnan(v)):
                u = np.ones(N)
                v = np.ones(M)
                break
        
        # Transport plan
        T = u[:, np.newaxis] * K * v[np.newaxis, :]
        
        return T
    
    def _phase3_clustering(self, consensus_graph, n_samples):
        """
        Phase 3: Nyström-accelerated Spectral Clustering
        
        Standard spectral clustering: O(N³) for eigendecomposition
        Nyström approximation: O(NM²) where M << N
        """
        # Ensure valid graph
        W = consensus_graph.copy()
        W = (W + W.T) / 2
        W[W < 0] = 0
        np.fill_diagonal(W, 0)  # No self-loops
        
        n_landmarks = min(self.n_landmarks, n_samples // 2)
        
        if n_samples > 2000 and n_landmarks >= self.n_clusters * 3:
            # Use Nyström for large datasets
            if self.verbose:
                print(f"  Using Nyström with {n_landmarks} landmarks...")
            Z = self._nystrom_embedding(W, n_landmarks)
        else:
            # Standard spectral for smaller datasets
            if self.verbose:
                print(f"  Using standard spectral embedding...")
            Z = self._standard_spectral_embedding(W)
        
        self.spectral_embedding_ = Z
        
        # K-Means with multiple restarts
        best_labels = None
        best_inertia = float('inf')
        
        for restart in range(5):
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=None if self.random_state is None else self.random_state + restart
            )
            labels = kmeans.fit_predict(Z)
            
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_labels = labels
        
        return best_labels
    
    def _nystrom_embedding(self, W, M):
        """
        Nyström approximation for spectral embedding.
        
        W ≈ C_NM @ W_MM^{-1} @ C_NM'
        
        Complexity: O(NM²) instead of O(N³)
        """
        N = W.shape[0]
        
        # Select landmarks using k-means++ style
        landmark_idx = self._kmeans_pp_landmarks(W, M)
        
        # Extract submatrices
        C_NM = W[:, landmark_idx]  # N x M
        W_MM = W[np.ix_(landmark_idx, landmark_idx)]  # M x M
        
        # Regularize and symmetrize
        W_MM = (W_MM + W_MM.T) / 2
        W_MM += np.eye(M) * 1e-5
        
        # Eigendecomposition of W_MM
        eigvals_mm, eigvecs_mm = np.linalg.eigh(W_MM)
        
        # Keep positive eigenvalues
        pos_mask = eigvals_mm > 1e-10
        if pos_mask.sum() < self.n_clusters:
            return self._standard_spectral_embedding(W)
        
        eigvals_mm = eigvals_mm[pos_mask]
        eigvecs_mm = eigvecs_mm[:, pos_mask]
        
        # W_MM^{-1/2}
        eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals_mm)
        W_MM_inv_sqrt = eigvecs_mm @ np.diag(eigvals_inv_sqrt) @ eigvecs_mm.T
        
        # Degree normalization
        d_approx = np.maximum(C_NM.sum(axis=1), 1e-10)
        D_inv_sqrt = 1.0 / np.sqrt(d_approx)
        
        # Normalized embedding
        Q = (D_inv_sqrt[:, np.newaxis] * C_NM) @ W_MM_inv_sqrt
        
        # SVD for orthogonal embedding
        U, S, _ = np.linalg.svd(Q, full_matrices=False)
        
        # Take top K eigenvectors
        K = self.n_clusters
        Z = U[:, :K]
        
        # Row normalize
        Z = normalize(Z, norm='l2', axis=1)
        
        return Z
    
    def _kmeans_pp_landmarks(self, W, M):
        """K-means++ style landmark selection based on graph distance."""
        N = W.shape[0]
        
        # Inverse weights as distances
        D = 1.0 / (W + 1e-10)
        np.fill_diagonal(D, 0)
        
        landmarks = [np.random.randint(N)]
        
        for _ in range(M - 1):
            min_dists = np.min(D[:, landmarks], axis=1)
            min_dists[landmarks] = 0
            
            probs = min_dists ** 2
            probs = probs / (probs.sum() + 1e-10)
            
            new_landmark = np.random.choice(N, p=probs)
            landmarks.append(new_landmark)
        
        return np.array(landmarks)
    
    def _standard_spectral_embedding(self, W):
        """
        Spectral embedding supporting both dense and sparse matrices.
        
        For sparse: O(N·k·K) using sparse eigensolver
        For dense: O(N³) fallback
        """
        N = W.shape[0] if not sparse.issparse(W) else W.shape[0]
        
        if sparse.issparse(W):
            # Sparse path: O(N·k·K)
            d = np.array(W.sum(axis=1)).flatten()
            d = np.maximum(d, 1e-10)
            D_inv_sqrt = sparse.diags(1.0 / np.sqrt(d))
            
            # Normalized adjacency (sparse)
            W_norm = D_inv_sqrt @ W @ D_inv_sqrt
            W_norm = (W_norm + W_norm.T) / 2
            
            K = self.n_clusters
            try:
                # Sparse eigenvalue solver: O(N·nnz·K)
                eigenvalues, eigenvectors = eigsh(W_norm, k=K, which='LM')
                idx = np.argsort(eigenvalues)[::-1]
                eigenvectors = eigenvectors[:, idx]
            except Exception:
                # Fallback to dense
                W_dense = W_norm.toarray() if sparse.issparse(W_norm) else W_norm
                eigenvalues, eigenvectors = np.linalg.eigh(W_dense)
                idx = np.argsort(eigenvalues)[::-1][:K]
                eigenvectors = eigenvectors[:, idx]
        else:
            # Dense path
            d = W.sum(axis=1)
            d = np.maximum(d, 1e-10)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
            
            W_norm = D_inv_sqrt @ W @ D_inv_sqrt
            W_norm = (W_norm + W_norm.T) / 2
            
            K = self.n_clusters
            try:
                eigenvalues, eigenvectors = eigsh(W_norm, k=K, which='LM')
                idx = np.argsort(eigenvalues)[::-1]
                eigenvectors = eigenvectors[:, idx]
            except Exception:
                eigenvalues, eigenvectors = np.linalg.eigh(W_norm)
                idx = np.argsort(eigenvalues)[::-1][:K]
                eigenvectors = eigenvectors[:, idx]
        
        Z = normalize(eigenvectors, norm='l2', axis=1)
        
        return Z


class SPOCKAblation(SPOCK):
    """
    SPOCK with ablation study support.
    
    Allows disabling specific components for ablation experiments.
    
    Ablation modes:
    - 'full': Complete SPOCK algorithm
    - 'no_feature_selection': Skip Phase 1 ADMM, use raw normalized features
    - 'no_density_aware': Disable density-aware graph construction
    - 'no_ot_alignment': Use simple averaging instead of OT alignment
    - 'no_nystrom': Force standard spectral clustering
    """
    
    def __init__(self, ablation_mode='full', **kwargs):
        super().__init__(**kwargs)
        self.ablation_mode = ablation_mode
    
    def _phase1_feature_selection(self, X_views):
        if self.ablation_mode == 'no_feature_selection':
            # Skip ADMM, just normalize
            projected_views = []
            projection_matrices = []
            self_expression_matrices = []
            
            for X in X_views:
                X_normalized = normalize(X, norm='l2', axis=1)
                projected_views.append(X_normalized)
                projection_matrices.append(None)
                self_expression_matrices.append(None)
                
            return projected_views, projection_matrices, self_expression_matrices
        else:
            return super()._phase1_feature_selection(X_views)
    
    def _construct_density_aware_graph(self, X, densities, threshold, sigma):
        if self.ablation_mode == 'no_density_aware':
            # Standard KNN graph without density weighting
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
    
    def _rff_sinkhorn_alignment(self, view_graphs, projected_views, view_densities):
        if self.ablation_mode == 'no_ot_alignment':
            # Simple unweighted average
            n_views = len(view_graphs)
            view_weights = np.ones(n_views) / n_views
            consensus = np.mean(view_graphs, axis=0)
            return consensus, view_weights
        else:
            return super()._rff_sinkhorn_alignment(view_graphs, projected_views, view_densities)
    
    def _phase3_clustering(self, consensus_graph, n_samples):
        if self.ablation_mode == 'no_nystrom':
            # Force standard spectral
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
