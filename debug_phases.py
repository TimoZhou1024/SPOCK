"""Debug script to analyze each phase of SPOCK."""

import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from spock.evaluation.metrics import evaluate_clustering

# Load data
data = loadmat('data/handwritten.mat')
X_cell = data['X']
X_views = [X_cell[0, i] for i in range(X_cell.shape[1])]
y_true = data['Y'].ravel()

print("=" * 60)
print("SPOCK Phase-by-Phase Debugging")
print("=" * 60)

# Test 1: Raw features directly
print("\n1. Raw concatenated features + KMeans:")
X_concat = np.hstack([normalize(X, 'l2') for X in X_views])
labels = KMeans(n_clusters=10, n_init=10, random_state=42).fit_predict(X_concat)
metrics = evaluate_clustering(y_true, labels)
print(f"   ACC: {metrics['ACC']:.4f}, NMI: {metrics['NMI']:.4f}")

# Test 2: Spectral clustering on single view
print("\n2. Spectral clustering on each view:")
for v, X in enumerate(X_views):
    X_norm = normalize(X, 'l2')
    try:
        sc = SpectralClustering(n_clusters=10, affinity='nearest_neighbors', n_neighbors=10, random_state=42)
        labels = sc.fit_predict(X_norm)
        metrics = evaluate_clustering(y_true, labels)
        print(f"   View {v+1}: ACC={metrics['ACC']:.4f}, NMI={metrics['NMI']:.4f}")
    except:
        print(f"   View {v+1}: FAILED")

# Test 3: SPOCK Phase 1 alone + KMeans
print("\n3. SPOCK Phase 1 (feature selection) + KMeans:")
from spock import SPOCK
spock = SPOCK(n_clusters=10, verbose=True, random_state=42)

# Manually run Phase 1
projected_views, P_list, S_list = spock._phase1_feature_selection(X_views)

# Cluster on projected views
X_proj_concat = np.hstack(projected_views)
labels = KMeans(n_clusters=10, n_init=10, random_state=42).fit_predict(X_proj_concat)
metrics = evaluate_clustering(y_true, labels)
print(f"   Phase 1 output + KMeans: ACC={metrics['ACC']:.4f}, NMI={metrics['NMI']:.4f}")

# Test 4: SPOCK Phase 1+2 (graph construction) + Spectral
print("\n4. SPOCK Phase 1+2 (feature selection + graph) + Spectral:")
consensus, weights = spock._phase2_graph_alignment(projected_views)

# Check graph quality
print(f"   Consensus graph stats:")
print(f"     - Min: {consensus.min():.6f}")
print(f"     - Max: {consensus.max():.6f}")
print(f"     - Mean: {consensus.mean():.6f}")
print(f"     - Sparsity: {(consensus > 0.01).sum() / consensus.size:.4f}")
print(f"     - View weights: {weights}")

# Test different spectral clustering approaches
from sklearn.cluster import SpectralClustering

# Approach 1: Direct sklearn SpectralClustering
print("\n   4a. sklearn SpectralClustering (precomputed):")
W = consensus.copy()
W = (W + W.T) / 2
W[W < 0] = 0
np.fill_diagonal(W, 0)

try:
    sc = SpectralClustering(n_clusters=10, affinity='precomputed', random_state=42, n_init=10)
    labels = sc.fit_predict(W)
    metrics = evaluate_clustering(y_true, labels)
    print(f"       ACC={metrics['ACC']:.4f}, NMI={metrics['NMI']:.4f}")
except Exception as e:
    print(f"       FAILED: {e}")

# Approach 2: Manual spectral with normalized Laplacian
print("\n   4b. Manual spectral (normalized Laplacian):")
d = W.sum(axis=1)
d = np.maximum(d, 1e-10)
D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
W_norm = D_inv_sqrt @ W @ D_inv_sqrt

try:
    eigenvalues, eigenvectors = np.linalg.eigh(W_norm)
    idx = np.argsort(eigenvalues)[::-1][:10]
    Z = eigenvectors[:, idx]
    Z = normalize(Z, 'l2', axis=1)
    labels = KMeans(n_clusters=10, n_init=10, random_state=42).fit_predict(Z)
    metrics = evaluate_clustering(y_true, labels)
    print(f"       ACC={metrics['ACC']:.4f}, NMI={metrics['NMI']:.4f}")
except Exception as e:
    print(f"       FAILED: {e}")

# Approach 3: Use Phase 1 projected features directly with spectral
print("\n   4c. Spectral on each projected view:")
for v, X_proj in enumerate(projected_views):
    try:
        sc = SpectralClustering(n_clusters=10, affinity='nearest_neighbors', n_neighbors=10, random_state=42)
        labels = sc.fit_predict(X_proj)
        metrics = evaluate_clustering(y_true, labels)
        print(f"       View {v+1}: ACC={metrics['ACC']:.4f}, NMI={metrics['NMI']:.4f}")
    except:
        print(f"       View {v+1}: FAILED")

# Approach 4: Build graphs from each view and check their quality
print("\n   4d. Individual view graphs:")
view_graphs = []
for v, X_proj in enumerate(projected_views):
    nbrs = NearestNeighbors(n_neighbors=11, algorithm='auto').fit(X_proj)
    distances, indices = nbrs.kneighbors(X_proj)
    sigma = np.median(distances[:, 1:]) + 1e-10
    
    N = X_proj.shape[0]
    G = np.zeros((N, N))
    for i in range(N):
        for j_idx, (idx, dist) in enumerate(zip(indices[i, 1:], distances[i, 1:])):
            G[i, idx] = np.exp(-dist**2 / (2 * sigma**2))
    G = (G + G.T) / 2
    view_graphs.append(G)
    
    print(f"       View {v+1}: mean={G.mean():.6f}, max={G.max():.6f}")

# Approach 5: Simple average of view graphs
print("\n   4e. Simple average of view graphs + spectral:")
avg_graph = np.mean(view_graphs, axis=0)
print(f"       Avg graph: mean={avg_graph.mean():.6f}, max={avg_graph.max():.6f}")

sc = SpectralClustering(n_clusters=10, affinity='precomputed', random_state=42, n_init=10)
labels = sc.fit_predict(avg_graph)
metrics = evaluate_clustering(y_true, labels)
print(f"       ACC={metrics['ACC']:.4f}, NMI={metrics['NMI']:.4f}")

# Approach 6: Weighted average based on view performance
print("\n   4f. Best single view + spectral (View 5):")
sc = SpectralClustering(n_clusters=10, affinity='precomputed', random_state=42, n_init=10)
labels = sc.fit_predict(view_graphs[4])  # View 5
metrics = evaluate_clustering(y_true, labels)
print(f"       ACC={metrics['ACC']:.4f}, NMI={metrics['NMI']:.4f}")

# Test 5: Full SPOCK
print("\n5. Full SPOCK:")
spock2 = SPOCK(n_clusters=10, verbose=False, random_state=42)
labels = spock2.fit_predict(X_views)
metrics = evaluate_clustering(y_true, labels)
print(f"   Full SPOCK: ACC={metrics['ACC']:.4f}, NMI={metrics['NMI']:.4f}")

print("\n" + "=" * 60)
