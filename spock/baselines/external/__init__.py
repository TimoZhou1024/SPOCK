"""
External Methods Directory

This directory contains cloned external multi-view clustering implementations.
These methods require git clone and additional dependencies (typically PyTorch).

================================================================================
QUICK SETUP
================================================================================

# Clone SCMVC (IEEE TMM 2024)
cd spock/baselines/external
git clone https://github.com/SongwuJob/SCMVC.git

# Install PyTorch (required for deep methods)
pip install torch>=1.12.0

================================================================================
AVAILABLE EXTERNAL METHODS
================================================================================

SCMVC - Self-Weighted Contrastive Fusion for Deep MVC
    Paper: IEEE Transactions on Multimedia (TMM) 2024
    Setup: git clone https://github.com/SongwuJob/SCMVC.git
    Requirements: torch>=1.12.0, scikit-learn

================================================================================
USAGE
================================================================================

After cloning, you can use the methods in experiments:

    # Method 1: Via get_baseline_methods
    from spock.baselines import get_baseline_methods
    methods = get_baseline_methods(
        n_clusters=10,
        include_scalable=True,
        include_external=True
    )

    # Method 2: Direct import
    from spock.baselines import SCMVCWrapper
    model = SCMVCWrapper(n_clusters=10, epochs=200)
    labels = model.fit_predict(X_views)

    # Check availability
    from spock.baselines import check_scalable_methods_availability
    print(check_scalable_methods_availability())
    # Output: {..., 'SCMVC': True/False, ...}

    # List setup instructions
    from spock.baselines import list_external_methods
    print(list_external_methods())

================================================================================
ADDING NEW EXTERNAL METHODS
================================================================================

1. Clone the repository:
   git clone https://github.com/<org>/<repo>.git <method_name>

2. Add a wrapper class in scalable_methods.py:

   class NewMethodWrapper(BaseExternalMVC):
       def __init__(self, n_clusters, ...):
           super().__init__(n_clusters, device='auto', random_state=None)
           self.name = 'NewMethod'

       def _get_external_paths(self):
           return ['NewMethod', 'newmethod']  # Possible directory names

       def _run_external(self, X_views):
           # Import and run external code
           from network import Network
           ...
           return labels

       def _run_fallback(self, X_views):
           # Fallback when external code not available
           ...

3. Register in EXTERNAL_METHODS dict

4. Update list_external_methods() with setup info

================================================================================
FALLBACK BEHAVIOR
================================================================================

If external code is not found or fails to run, all wrappers will automatically
fall back to a simplified implementation or basic K-Means clustering.
This ensures experiments can still run (with reduced accuracy) even without
the external dependencies.
"""
