"""
SPOCK: Scalable and Structure-Preserving Optimal Transport based Clustering 
with Kernel-density-estimation for Imperfect Multi-View Data
"""

from .core import SPOCK, SPOCKAblation
from .evaluation import evaluate_clustering

__version__ = "1.0.0"

__all__ = ['SPOCK', 'SPOCKAblation', 'evaluate_clustering']
