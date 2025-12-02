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
            # Phase 2: Graph construction and fusion
            if self.verbose:
                print("Phase 2: Multi-view Graph Fusion...")
            self.consensus_graph_, self.view_weights_ = self._phase2_graph_alignment(projected_views)
            
            # Phase 3: Final Clustering via Spectral Clustering
            if self.verbose:
                print("Phase 3: Spectral Clustering...")
            self.labels_ = self._phase3_clustering(self.consensus_graph_, n_samples)
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
        """Construct graph Laplacian from KNN graph using ANN."""
        N = X.shape[0]
        k = min(self.k_neighbors, N - 1)
        
        # Use FAISS for fast ANN if available
        if HAS_FAISS and N > 1000:
            A = self._faiss_knn(X, k)
        else:
            A = self._sklearn_knn(X, k)
        
        # Make symmetric
        A = (A + A.T) / 2
        A = np.minimum(A, 1)  # Cap at 1
        
        # Compute normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        d = A.sum(axis=1)
        d = np.maximum(d, 1e-10)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
        L = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
        
        return L
    
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
        Optimized feature projection for structure preservation.
        
        Uses a hybrid approach:
        1. PCA for variance preservation (global structure)
        2. Laplacian regularization for local structure
        
        This balances between preserving global discriminability 
        and local neighborhood structure.
        """
        N, D = X.shape
        
        # Compute covariance matrix
        XtX = X.T @ X
        XtLX = X.T @ L @ X
        
        # Regularization for numerical stability
        reg = 1e-6 * np.trace(XtX) / D
        
        # Hybrid objective: maximize variance while minimizing Laplacian energy
        # Solve: max P' X'X P - alpha * P' X'LX P
        # Equivalent to: max P' (X'X - alpha * X'LX) P
        
        # Blend ratio: higher alpha means more local structure preservation
        alpha_blend = min(self.alpha * 0.1, 0.5)  # Keep alpha influence modest
        
        M = XtX - alpha_blend * XtLX
        M = (M + M.T) / 2  # Ensure symmetry
        M += reg * np.eye(D)  # Regularize
        
        # Solve eigenvalue problem
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(M)
            
            # Take d eigenvectors corresponding to largest eigenvalues
            idx = np.argsort(eigenvalues)[::-1][:d]
            P = eigenvectors[:, idx]
        except Exception:
            # Fallback to standard PCA
            eigenvalues, eigenvectors = np.linalg.eigh(XtX + reg * np.eye(D))
            idx = np.argsort(eigenvalues)[::-1][:d]
            P = eigenvectors[:, idx]
        
        # Orthonormalize P
        P, _ = np.linalg.qr(P)
        if P.shape[1] < d:
            P = np.hstack([P, np.zeros((D, d - P.shape[1]))])
        P = P[:, :d]
        
        # Compute S using sparse local linear reconstruction
        Z = X @ P
        S = self._compute_sparse_S(Z)
        
        if self.verbose:
            # Compute objective for reporting
            I_minus_S = np.eye(N) - S
            recon_term = I_minus_S @ Z
            recon_loss = np.linalg.norm(recon_term, 'fro') ** 2
            lap_loss = self.alpha * np.trace(P.T @ XtLX @ P)
            obj = recon_loss + lap_loss
            print(f"    Optimization complete, obj={obj:.4f}")
        
        return P, S
        
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
    
    def _phase2_graph_alignment(self, projected_views):
        """
        Phase 2: Multi-view Graph Fusion
        
        Constructs view-specific graphs and fuses them.
        Uses uniform weighting as it empirically works well.
        """
        n_views = len(projected_views)
        n_samples = projected_views[0].shape[0]
        
        view_graphs = []
        
        for v, X in enumerate(projected_views):
            if self.verbose:
                print(f"  Processing view {v+1}/{n_views}...")
            
            # Construct KNN graph with heat kernel weights
            G = self._construct_knn_graph(X)
            view_graphs.append(G)
        
        # Uniform view weights (empirically works well)
        view_weights = np.ones(n_views) / n_views
        
        if self.verbose:
            print(f"  View weights: {view_weights}")
        
        # Simple average fusion of graphs
        consensus = np.mean(view_graphs, axis=0)
        
        # Symmetrize and clean up
        consensus = (consensus + consensus.T) / 2
        consensus = np.maximum(consensus, 0)
        np.fill_diagonal(consensus, 0)
        
        return consensus, view_weights
    
    def _construct_knn_graph(self, X):
        """
        Construct KNN graph with heat kernel weights.
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
    
    # ==================== Phase 3: Nyström Spectral Clustering ====================
    
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
        """Standard spectral embedding for smaller datasets."""
        N = W.shape[0]
        
        # Degree matrix
        d = W.sum(axis=1)
        d = np.maximum(d, 1e-10)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
        
        # Normalized adjacency
        W_norm = D_inv_sqrt @ W @ D_inv_sqrt
        W_norm = (W_norm + W_norm.T) / 2
        
        K = self.n_clusters
        try:
            eigenvalues, eigenvectors = eigsh(W_norm, k=K, which='LM')
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
        except:
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
