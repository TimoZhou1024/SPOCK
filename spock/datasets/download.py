"""
Dataset Download Utilities

Downloads real multi-view benchmark datasets from public sources.
"""

import os
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm


# Dataset URLs and metadata
# Sources: UCI ML Repository, Original academic sources

DATASET_URLS = {
    # Handwritten digits (UCI) - Multiple feature dataset
    'handwritten': {
        'url': 'https://archive.ics.uci.edu/static/public/72/multiple+features.zip',
        'type': 'zip',
        'description': 'UCI Multiple Features / Handwritten Digits (6 views, 2000 samples, 10 classes)',
    },
    
    # COIL-20 (Columbia Object Image Library)
    'coil20': {
        'url': 'https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip',
        'type': 'zip',
        'description': 'COIL-20 Object Images (multi-view capable, 1440 samples, 20 classes)',
    },
}

# Provide instructions for manual download
MANUAL_DOWNLOAD_INFO = {
    'bbcsport': {
        'description': 'BBC Sport (2 views, 544 samples, 5 classes)',
        'source': 'http://mlg.ucd.ie/datasets/segment.html',
        'instructions': '''
1. Visit: http://mlg.ucd.ie/datasets/segment.html
2. Download the BBCSport dataset
3. Save as: ./data/bbcsport.mat
Alternative: Search "BBC Sport multi-view dataset .mat" on GitHub
''',
    },
    'caltech101': {
        'description': 'Caltech101 (6 views, various samples, 7/20 classes)',
        'source': 'https://data.caltech.edu/records/mzrjq-6wc02',
        'instructions': '''
1. Visit: https://data.caltech.edu/records/mzrjq-6wc02
2. Or Kaggle: https://www.kaggle.com/datasets/imbikramsaha/caltech-101
3. Extract features using standard descriptors (Gabor, GIST, HOG, LBP, etc.)
4. Save processed .mat file as: ./data/Caltech101_7.mat or ./data/Caltech101_20.mat
''',
    },
    'reuters': {
        'description': 'Reuters Multilingual (5 views, 18758 samples, 6 classes)',
        'source': 'https://archive.ics.uci.edu/dataset/259',
        'instructions': '''
1. Visit: https://archive.ics.uci.edu/dataset/259/reuters+rcv1+rcv2+multilingual+multiview+text+categorization+test+collection
2. Download and preprocess to .mat format
3. Save as: ./data/Reuters.mat
''',
    },
    'nuswide': {
        'description': 'NUS-WIDE Object (5 views, 30000 samples, 31 classes)',
        'source': 'https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html',
        'instructions': '''
1. Visit: https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html
2. Download low-level features
3. Process and save as: ./data/NUSWide.mat
''',
    },
    '3sources': {
        'description': '3Sources News (3 views, 169 samples, 6 classes)',
        'source': 'http://mlg.ucd.ie/datasets/3sources.html',
        'instructions': '''
1. Visit: http://mlg.ucd.ie/datasets/3sources.html
2. Download the dataset
3. Save as: ./data/3sources.mat
''',
    },
    'scene15': {
        'description': 'Scene15 (3 views, 4485 samples, 15 classes)',
        'source': 'https://figshare.com/articles/dataset/Scene-15_dataset/7007177',
        'instructions': '''
1. Search "Scene-15 multi-view dataset" on academic repositories
2. Download and process feature views
3. Save as: ./data/Scene15.mat
''',
    },
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path, desc=None):
    """Download a file with progress bar."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if desc is None:
        desc = output_path.name
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_dataset(name, data_dir='./data', force=False):
    """
    Download a specific dataset.
    
    Parameters
    ----------
    name : str
        Dataset name. Use list_available_datasets() to see options.
    data_dir : str
        Directory to save the dataset.
    force : bool
        If True, re-download even if file exists.
        
    Returns
    -------
    str
        Path to the downloaded dataset file.
    """
    name_lower = name.lower().replace('-', '_').replace(' ', '_')
    
    if name_lower not in DATASET_URLS:
        available = list(DATASET_URLS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    
    info = DATASET_URLS[name_lower]
    url = info['url']
    file_type = info['type']
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename
    if file_type == 'mat':
        output_file = data_dir / f"{name_lower}.mat"
    elif file_type == 'tar':
        output_file = data_dir / f"{name_lower}.tar"
    elif file_type == 'zip':
        output_file = data_dir / f"{name_lower}.zip"
    else:
        output_file = data_dir / url.split('/')[-1]
    
    # Check if already exists
    if output_file.exists() and not force:
        print(f"Dataset {name} already exists at {output_file}")
        return str(output_file)
    
    print(f"Downloading {info['description']}...")
    success = download_file(url, output_file, desc=name)
    
    if not success:
        raise RuntimeError(f"Failed to download {name}")
    
    # Extract if needed
    if file_type == 'tar':
        print(f"Extracting {output_file}...")
        with tarfile.open(output_file, 'r') as tar:
            tar.extractall(data_dir)
    elif file_type == 'zip':
        print(f"Extracting {output_file}...")
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        # Convert UCI mfeat to .mat format if needed
        if name_lower == 'handwritten':
            _convert_uci_mfeat(data_dir)
    
    print(f"Dataset {name} downloaded successfully to {output_file}")
    return str(output_file)


def _convert_uci_mfeat(data_dir):
    """Convert UCI mfeat files to a single .mat file."""
    import numpy as np
    from scipy.io import savemat
    
    data_dir = Path(data_dir)
    
    # Find the mfeat directory (could be nested)
    mfeat_dir = None
    for pattern in ['mfeat', 'multiple+features', 'multiple-features']:
        candidate = data_dir / pattern
        if candidate.exists():
            mfeat_dir = candidate
            break
    
    if mfeat_dir is None:
        # Try looking for mfeat-* files directly
        if (data_dir / 'mfeat-fac').exists():
            mfeat_dir = data_dir
        else:
            print("  Could not find mfeat feature files")
            return
    
    # UCI mfeat feature files
    feature_files = {
        'fac': 'mfeat-fac',   # 216 profile correlations
        'fou': 'mfeat-fou',   # 76 Fourier coefficients
        'kar': 'mfeat-kar',   # 64 Karhunen-Love coefficients
        'mor': 'mfeat-mor',   # 6 morphological features
        'pix': 'mfeat-pix',   # 240 pixel averages
        'zer': 'mfeat-zer',   # 47 Zernike moments
    }
    
    views = []
    for name, filename in feature_files.items():
        filepath = mfeat_dir / filename
        if filepath.exists():
            data = np.loadtxt(filepath)
            views.append(data)
            print(f"  Loaded {name}: shape {data.shape}")
    
    if len(views) == 6:
        # Labels: 200 samples per class, 10 classes
        labels = np.repeat(np.arange(10), 200)
        
        # Save as .mat - use cell array format
        # Create an object array that can hold different shaped matrices
        X = np.empty((6,), dtype=object)
        for i, v in enumerate(views):
            X[i] = v
        
        mat_data = {
            'X': X,
            'Y': labels.reshape(-1, 1),
        }
        
        output_file = data_dir / 'handwritten.mat'
        savemat(output_file, mat_data)
        print(f"  Converted to {output_file}")
    else:
        print(f"  Warning: Expected 6 views, found {len(views)}")


def download_all(data_dir='./data', force=False):
    """
    Download all available datasets.
    
    Parameters
    ----------
    data_dir : str
        Directory to save datasets.
    force : bool
        If True, re-download even if files exist.
    """
    print("Downloading all datasets...\n")
    
    for name in DATASET_URLS:
        try:
            download_dataset(name, data_dir, force)
            print()
        except Exception as e:
            print(f"Error downloading {name}: {e}\n")
    
    print("Done!")


def list_available_datasets():
    """List all available datasets with descriptions."""
    print("\n" + "=" * 70)
    print("Available Multi-View Datasets")
    print("=" * 70)
    
    print("\n[Auto-Download Available]")
    for name, info in DATASET_URLS.items():
        print(f"  • {name:15s} - {info['description']}")
    
    print("\n[Manual Download Required]")
    for name, info in MANUAL_DOWNLOAD_INFO.items():
        print(f"  • {name:15s} - {info['description']}")
    
    print("\n" + "=" * 70)
    print("Usage: python -m spock.datasets.download --dataset <name>")
    print("       python -m spock.datasets.download --manual <name>")
    print("=" * 70 + "\n")


def show_manual_instructions(name):
    """Show manual download instructions for a dataset."""
    name_lower = name.lower().replace('-', '_').replace(' ', '_')
    
    if name_lower not in MANUAL_DOWNLOAD_INFO:
        print(f"No manual download info for '{name}'")
        print(f"Available: {list(MANUAL_DOWNLOAD_INFO.keys())}")
        return
    
    info = MANUAL_DOWNLOAD_INFO[name_lower]
    print("\n" + "=" * 70)
    print(f"Manual Download Instructions: {name}")
    print("=" * 70)
    print(f"\nDescription: {info['description']}")
    print(f"Source: {info['source']}")
    print(f"\nInstructions:{info['instructions']}")
    print("=" * 70 + "\n")


def check_downloaded(data_dir='./data'):
    """Check which datasets have been downloaded."""
    data_dir = Path(data_dir)
    
    print("\nDataset Status:")
    print("=" * 60)
    
    # Check auto-download datasets
    all_datasets = list(DATASET_URLS.keys())
    
    # Also check for common manual datasets
    manual_files = {
        'bbcsport': 'bbcsport.mat',
        'caltech101_7': 'Caltech101_7.mat', 
        'caltech101_20': 'Caltech101_20.mat',
        'scene15': 'Scene15.mat',
        'reuters': 'Reuters.mat',
        'nuswide': 'NUSWide.mat',
        '3sources': '3sources.mat',
        'msrcv1': 'MSRCV1.mat',
    }
    
    for name in all_datasets:
        mat_file = data_dir / f"{name}.mat"
        if mat_file.exists():
            size_mb = mat_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ {name:15s} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {name:15s} (not downloaded)")
    
    print("-" * 60)
    print("Manual datasets:")
    for name, filename in manual_files.items():
        mat_file = data_dir / filename
        if mat_file.exists():
            size_mb = mat_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ {name:15s} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {name:15s} (use --manual {name})")
    
    print("=" * 60 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download multi-view datasets')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name to download (or "all" for all datasets)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to save datasets')
    parser.add_argument('--list', action='store_true',
                        help='List available datasets')
    parser.add_argument('--status', action='store_true',
                        help='Check download status')
    parser.add_argument('--manual', type=str, default=None,
                        help='Show manual download instructions for a dataset')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets()
    elif args.status:
        check_downloaded(args.data_dir)
    elif args.manual:
        show_manual_instructions(args.manual)
    elif args.dataset:
        if args.dataset.lower() == 'all':
            download_all(args.data_dir, args.force)
        else:
            download_dataset(args.dataset, args.data_dir, args.force)
    else:
        parser.print_help()
