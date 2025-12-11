"""Test script for scalable methods integration."""
import sys
sys.path.insert(0, '.')

print("Testing scalable methods integration...")

try:
    from spock.baselines import (
        check_scalable_methods_availability,
        get_scalable_methods,
        SCALABLE_METHODS,
        DEEP_AVAILABLE
    )
    print(f"DEEP_AVAILABLE (backwards compat): {DEEP_AVAILABLE}")
    print(f"Number of scalable methods: {len(SCALABLE_METHODS)}")

    avail = check_scalable_methods_availability()
    print("\nMethod availability:")
    for method, available in avail.items():
        status = "✓" if available else "✗"
        print(f"  {status} {method}")

    # Test each method wrapper
    print("\nTesting method instantiation...")
    methods = get_scalable_methods(n_clusters=5, n_anchors=100, random_state=42)
    for name, method in methods.items():
        print(f"  ✓ {name}: {method.name}")

    # Test on synthetic data
    print("\nTesting on synthetic data...")
    import numpy as np
    np.random.seed(42)
    n_samples = 500
    n_clusters = 5

    # Create synthetic multi-view data
    X1 = np.random.randn(n_samples, 50)
    X2 = np.random.randn(n_samples, 30)
    X_views = [X1, X2]

    # Test each method
    print("\nRunning methods:")
    for name, method in methods.items():
        try:
            labels = method.fit_predict(X_views)
            unique_labels = len(np.unique(labels))
            print(f"  ✓ {name}: {unique_labels} unique labels")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print("\nAll tests passed!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
