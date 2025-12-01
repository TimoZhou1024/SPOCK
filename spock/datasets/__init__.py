"""
SPOCK Datasets Module
"""
from .loaders import (
    MultiViewDataset,
    load_dataset,
    load_handwritten,
    load_caltech101,
    load_scene15,
    load_reuters,
    load_bbcsport,
    load_msrcv1,
    load_nuswide,
    get_available_datasets
)

from .download import (
    download_dataset,
    download_all,
    list_available_datasets,
    check_downloaded,
    DATASET_URLS,
)

__all__ = [
    'MultiViewDataset',
    'load_dataset',
    'load_handwritten',
    'load_caltech101',
    'load_scene15',
    'load_reuters',
    'load_bbcsport',
    'load_msrcv1',
    'load_nuswide',
    'get_available_datasets',
    # Download utilities
    'download_dataset',
    'download_all',
    'list_available_datasets',
    'check_downloaded',
    'DATASET_URLS',
]
