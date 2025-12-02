"""
Test SPOCK scalability to verify near-linear time complexity.
"""
import numpy as np
import time
from spock import SPOCK

def generate_synthetic_data(n_samples, n_views=6, n_clusters=10, dim=100):
    """Generate synthetic multi-view data."""
    np.random.seed(42)
    
    # Generate cluster centers
    centers = np.random.randn(n_clusters, dim) * 5
    
    # Generate data
    views = []
    labels = np.repeat(np.arange(n_clusters), n_samples // n_clusters)
    
    for v in range(n_views):
        X = np.zeros((n_samples, dim))
        for i in range(n_samples):
            c = labels[i]
            X[i] = centers[c] + np.random.randn(dim) * (0.5 + 0.3 * v)
        
        # Add view-specific transformation
        transform = np.random.randn(dim, dim) * 0.1 + np.eye(dim)
        X = X @ transform
        views.append(X)
    
    return views, labels

def test_scalability():
    print("=" * 60)
    print("SPOCK Scalability Test")
    print("=" * 60)
    
    sample_sizes = [500, 1000, 2000, 4000, 8000]
    results = []
    
    for n in sample_sizes:
        print(f"\nTesting N = {n}...")
        
        # Generate data
        views, labels = generate_synthetic_data(n)
        
        # Test Spectral mode
        spock_spectral = SPOCK(
            n_clusters=10,
            use_spectral=True,
            verbose=False,
            random_state=42
        )
        
        start = time.time()
        spock_spectral.fit(views)
        time_spectral = time.time() - start
        
        # Test KMeans mode
        spock_kmeans = SPOCK(
            n_clusters=10,
            use_spectral=False,
            verbose=False,
            random_state=42
        )
        
        start = time.time()
        spock_kmeans.fit(views)
        time_kmeans = time.time() - start
        
        # Compute accuracy
        from spock.evaluation import clustering_accuracy
        acc_spectral = clustering_accuracy(labels, spock_spectral.labels_)
        acc_kmeans = clustering_accuracy(labels, spock_kmeans.labels_)
        
        results.append({
            'N': n,
            'time_spectral': time_spectral,
            'time_kmeans': time_kmeans,
            'acc_spectral': acc_spectral,
            'acc_kmeans': acc_kmeans
        })
        
        print(f"  Spectral: {time_spectral:.2f}s (ACC={acc_spectral:.3f})")
        print(f"  KMeans:   {time_kmeans:.2f}s (ACC={acc_kmeans:.3f})")
    
    # Analyze complexity
    print("\n" + "=" * 60)
    print("Complexity Analysis")
    print("=" * 60)
    print(f"{'N':>8} {'T_spec':>10} {'T_km':>10} {'T_spec/N':>12} {'T_km/N':>12}")
    print("-" * 60)
    
    for r in results:
        t_per_n_spec = r['time_spectral'] / r['N'] * 1000  # ms per sample
        t_per_n_km = r['time_kmeans'] / r['N'] * 1000
        print(f"{r['N']:>8} {r['time_spectral']:>10.2f}s {r['time_kmeans']:>10.2f}s "
              f"{t_per_n_spec:>10.3f}ms {t_per_n_km:>10.3f}ms")
    
    # Check near-linear scaling
    print("\n" + "=" * 60)
    print("Scaling Factor (relative to N=500)")
    print("=" * 60)
    
    base_time_spec = results[0]['time_spectral']
    base_time_km = results[0]['time_kmeans']
    base_n = results[0]['N']
    
    print(f"{'N':>8} {'N_ratio':>10} {'T_ratio_spec':>12} {'T_ratio_km':>12} {'Expected O(N)':>14}")
    print("-" * 60)
    
    for r in results:
        n_ratio = r['N'] / base_n
        t_ratio_spec = r['time_spectral'] / base_time_spec
        t_ratio_km = r['time_kmeans'] / base_time_km
        print(f"{r['N']:>8} {n_ratio:>10.1f}x {t_ratio_spec:>12.1f}x {t_ratio_km:>12.1f}x {n_ratio:>14.1f}x")
    
    # Verdict
    print("\n" + "=" * 60)
    last = results[-1]
    first = results[0]
    n_ratio = last['N'] / first['N']
    t_ratio_spec = last['time_spectral'] / first['time_spectral']
    t_ratio_km = last['time_kmeans'] / first['time_kmeans']
    
    # Near-linear: T(16N) should be ~16-32 times T(N) (allowing log factor)
    spec_complexity = "O(N log N)" if t_ratio_spec < n_ratio * 3 else "O(N²)"
    km_complexity = "O(N log N)" if t_ratio_km < n_ratio * 3 else "O(N²)"
    
    print(f"Spectral mode complexity: ~{spec_complexity}")
    print(f"KMeans mode complexity:   ~{km_complexity}")
    print("=" * 60)

if __name__ == "__main__":
    test_scalability()
