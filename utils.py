"""
============================================================================
Utility Functions — Fruit Sugar Content Estimation
============================================================================
Common helper functions used across the entire project.
============================================================================
"""

import os
import random
import numpy as np

# PyTorch is optional for dataset setup (only needed for training)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Ensures the same results every time you run the code,
    which is crucial for scientific experiments and debugging.
    
    Args:
        seed (int): Random seed value (default: 42)
    """
    random.seed(seed)               # Python's built-in random
    np.random.seed(seed)             # NumPy random
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)          # PyTorch CPU random
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # PyTorch GPU random
            # Make CUDA operations deterministic (slightly slower)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to {seed} for reproducibility")


def get_device():
    """
    Automatically detect and return the best available device.
    
    Returns:
        torch.device: 'cuda' if GPU available, else 'cpu'
    """
    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch not installed — install with: pip install torch torchvision")
        return None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"✅ Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU detected — using CPU (training will be slower)")
    return device


def ensure_dir(path):
    """
    Create a directory (and parents) if it doesn't exist.
    
    Args:
        path (str): Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def count_files(directory, extensions=None):
    """
    Count total files in a directory (recursively).
    
    Args:
        directory (str): Root directory to scan
        extensions (set): File extensions to count (e.g., {'.jpg', '.png'})
    
    Returns:
        int: Total file count
    """
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    count = 0
    for root, dirs, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in extensions:
                count += 1
    return count


def format_time(seconds):
    """Convert seconds to a human-readable string (e.g., '2m 35s')."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def print_separator(title="", char="═", width=70):
    """Print a formatted section separator for clean console output."""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * width)


def print_dataset_summary(train_count, val_count, test_count, num_classes):
    """Print a formatted summary of the dataset split."""
    total = train_count + val_count + test_count
    print_separator("Dataset Summary")
    print(f"  Total images:    {total}")
    print(f"  Training set:    {train_count} ({train_count/total*100:.1f}%)")
    print(f"  Validation set:  {val_count} ({val_count/total*100:.1f}%)")
    print(f"  Test set:        {test_count} ({test_count/total*100:.1f}%)")
    print(f"  Number of classes: {num_classes}")
    print_separator()
