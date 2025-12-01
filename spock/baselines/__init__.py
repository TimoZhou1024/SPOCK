"""
SPOCK Baselines Module
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
    get_baseline_methods
)

__all__ = [
    'BaseMultiViewClustering',
    'ConcatKMeans',
    'ConcatSpectral',
    'BestViewSpectral',
    'CoRegSpectral',
    'MultiViewKMeans',
    'MVSC',
    'LMvSC',
    'MLAN',
    'get_baseline_methods'
]
