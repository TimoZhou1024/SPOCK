"""
SPOCK Baselines Module

This module provides baseline methods for multi-view clustering comparison.

Traditional Methods (always available):
    - ConcatKMeans, ConcatSpectral, BestViewSpectral
    - CoRegSpectral, MultiViewKMeans, MVSC, LMvSC, MLAN

Scalable Methods (near-linear complexity, always available):
    - LMVSC: Large-scale Multi-View Spectral Clustering (AAAI 2020)
    - SMVSC: Scalable Multi-View Subspace Clustering (ACM MM 2021)
    - FMCNOF: Fast Multi-view Clustering via NMF (TIP 2021)
    - EOMSC-CA: Efficient One-pass Multi-view Subspace Clustering (AAAI 2022)
    - BMVC: Binary Multi-View Clustering (TPAMI 2019)
    - FastMVC: Fast Multi-View Clustering (late fusion)

External Methods (require git clone):
    - SCMVC: Self-Weighted Contrastive Fusion (IEEE TMM 2024)

Usage:
    # Traditional methods only
    methods = get_baseline_methods(n_clusters)

    # Include scalable SOTA methods
    methods = get_baseline_methods(n_clusters, include_scalable=True)

    # Include external methods (need git clone first)
    methods = get_baseline_methods(n_clusters, include_scalable=True, include_external=True)

    # Only scalable methods for large-scale comparison
    methods = get_scalable_methods(n_clusters)

    # List external methods setup instructions
    from spock.baselines import list_external_methods
    print(list_external_methods())
"""

from .methods import (
    BaseMultiViewClustering,
    ConcatKMeans,
    ConcatSpectral,
    BestViewSpectral,
    CoRegSpectral,
    MultiViewKMeans,
    MVSC,
    LMvSC,
    MLAN,
    get_baseline_methods as _get_traditional_methods
)

# Import scalable methods (always available - pure Python/NumPy)
from .scalable_methods import (
    BaseScalableMVC,
    BaseExternalMVC,
    LMVSCWrapper,
    SMVSCWrapper,
    FMCNOFWrapper,
    EOMSCCAWrapper,
    BMVCWrapper,
    FastMVCWrapper,
    SCMVCWrapper,
    EFIMVCWrapper,
    ALPCWrapper,
    get_scalable_methods,
    check_scalable_methods_availability,
    list_external_methods,
    SCALABLE_METHODS,
    EXTERNAL_METHODS,
)

# For backwards compatibility
DEEP_AVAILABLE = True  # Scalable methods are always available
DEEP_METHODS = {**SCALABLE_METHODS, **EXTERNAL_METHODS}
get_deep_methods = get_scalable_methods
check_deep_methods_availability = check_scalable_methods_availability


def get_baseline_methods(n_clusters, random_state=None, include_deep=False,
                         include_scalable=False, include_external=False, n_anchors=500):
    """
    Get all baseline methods for comparison.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    random_state : int, optional
        Random seed for reproducibility.
    include_deep : bool, default=False
        Deprecated alias for include_scalable. Use include_scalable instead.
    include_scalable : bool, default=False
        Whether to include scalable SOTA methods (LMVSC, SMVSC, etc.).
    include_external : bool, default=False
        Whether to include external methods (SCMVC, etc.). Requires git clone.
    n_anchors : int, default=500
        Number of anchors for anchor-based methods.

    Returns
    -------
    methods : dict
        Dictionary of method name -> method instance.
    """
    # Get traditional methods
    methods = _get_traditional_methods(n_clusters, random_state)

    # Add scalable methods if requested
    if include_scalable or include_deep:
        scalable = get_scalable_methods(
            n_clusters,
            n_anchors=n_anchors,
            random_state=random_state,
            include_external=include_external
        )
        methods.update(scalable)

    return methods


__all__ = [
    # Base classes
    'BaseMultiViewClustering',
    'BaseScalableMVC',
    'BaseExternalMVC',
    # Traditional methods
    'ConcatKMeans',
    'ConcatSpectral',
    'BestViewSpectral',
    'CoRegSpectral',
    'MultiViewKMeans',
    'MVSC',
    'LMvSC',
    'MLAN',
    # Scalable methods (built-in)
    'LMVSCWrapper',
    'SMVSCWrapper',
    'FMCNOFWrapper',
    'EOMSCCAWrapper',
    'BMVCWrapper',
    'FastMVCWrapper',
    'SCALABLE_METHODS',
    # External methods
    'SCMVCWrapper',
    'EFIMVCWrapper',
    'ALPCWrapper',
    'EXTERNAL_METHODS',
    'list_external_methods',
    # Factory functions
    'get_baseline_methods',
    'get_scalable_methods',
    'check_scalable_methods_availability',
    # Backwards compatibility
    'DEEP_AVAILABLE',
    'DEEP_METHODS',
    'get_deep_methods',
    'check_deep_methods_availability',
]
