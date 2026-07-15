"""
============================================================================
Dataset Utilities — Fruit Sugar Content Estimation
============================================================================
Handles dataset loading, augmentation, and DataLoader creation.

This module provides:
  - FruitSugarDataset: Custom PyTorch Dataset class
  - get_transforms():  Training and validation image transforms
  - get_dataloaders():  Ready-to-use DataLoaders for training
============================================================================
"""

import os
import sys
from collections import Counter

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# Import project configuration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ORGANIZED_DIR, SUGAR_CLASSES, NUM_CLASSES, IMG_SIZE,
    BATCH_SIZE, NUM_WORKERS, IMAGE_EXTENSIONS,
    AUGMENTATION, IMAGENET_MEAN, IMAGENET_STD
)


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════════
class FruitSugarDataset(Dataset):
    """
    Custom PyTorch Dataset for fruit sugar content estimation.
    
    Loads images from the organized directory structure:
        organized_data/{split}/{class_name}/image.jpg
    
    Each image is labeled with a sugar content class:
        0 = Low_Sugar, 1 = Medium_Sugar, 2 = High_Sugar
    
    Args:
        root_dir (str):       Path to split directory (e.g., organized_data/train)
        transform (callable): Image transformations to apply
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Build list of (image_path, label_index, fruit_name) tuples
        self.samples = []
        self.class_counts = Counter()
        
        # Iterate through class directories
        for class_idx, class_name in enumerate(SUGAR_CLASSES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"  ⚠️  Class directory not found: {class_dir}")
                continue
            
            for filename in os.listdir(class_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue
                
                filepath = os.path.join(class_dir, filename)
                
                # Extract fruit name from filename (format: fruit_originalname.ext)
                fruit_name = filename.split("_")[0] if "_" in filename else "unknown"
                
                self.samples.append((filepath, class_idx, fruit_name))
                self.class_counts[class_name] += 1
        
        print(f"  📂 Loaded {len(self.samples)} images from {root_dir}")
        for cls, count in sorted(self.class_counts.items()):
            print(f"     {cls}: {count}")
    
    def __len__(self):
        """Return total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Load and return one sample.
        
        Args:
            idx (int): Sample index
        
        Returns:
            tuple: (image_tensor, label_index)
        """
        filepath, label, fruit_name = self.samples[idx]
        
        # Load image as RGB (handles grayscale, RGBA, etc.)
        try:
            image = Image.open(filepath).convert("RGB")
        except Exception as e:
            print(f"  ❌ Error loading {filepath}: {e}")
            # Return a blank image if loading fails
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        
        # Apply transforms (resize, augment, normalize)
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_fruit_name(self, idx):
        """Get the fruit name for a given sample index."""
        return self.samples[idx][2]
    
    def get_filepath(self, idx):
        """Get the file path for a given sample index."""
        return self.samples[idx][0]


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════
def get_transforms(is_training=True):
    """
    Get image transformation pipeline.
    
    Training transforms include data augmentation (random flips, rotations,
    color jitter) to make the model more robust.
    
    Validation/test transforms only resize and normalize — no augmentation,
    because we want consistent evaluation.
    
    Args:
        is_training (bool): If True, include augmentation transforms
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if is_training:
        # Training: augmentation + normalization
        return transforms.Compose([
            # Resize to slightly larger, then random crop for variety
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            
            # Random horizontal flip (50% chance)
            transforms.RandomHorizontalFlip(p=AUGMENTATION["horizontal_flip_prob"]),
            
            # Random vertical flip (20% chance — fruits can be any orientation)
            transforms.RandomVerticalFlip(p=AUGMENTATION["vertical_flip_prob"]),
            
            # Small random rotation (±15°)
            transforms.RandomRotation(degrees=AUGMENTATION["rotation_degrees"]),
            
            # Random color variations (brightness, contrast, saturation, hue)
            # This is CRITICAL for our task because we rely on color features
            transforms.ColorJitter(
                brightness=AUGMENTATION["color_jitter_brightness"],
                contrast=AUGMENTATION["color_jitter_contrast"],
                saturation=AUGMENTATION["color_jitter_saturation"],
                hue=AUGMENTATION["color_jitter_hue"],
            ),
            
            # Convert PIL Image to PyTorch Tensor (0-255 → 0.0-1.0)
            transforms.ToTensor(),
            
            # Normalize with ImageNet statistics (required for pre-trained models)
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        # Validation/Test: no augmentation, just resize + normalize
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ═══════════════════════════════════════════════════════════════════════════
# DATALOADER CREATION
# ═══════════════════════════════════════════════════════════════════════════
def get_dataloaders(organized_dir=None, batch_size=None, use_weighted_sampler=True):
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        organized_dir (str): Path to organized dataset (default: from config)
        batch_size (int): Batch size (default: from config)
        use_weighted_sampler (bool): If True, oversample minority classes
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if organized_dir is None:
        organized_dir = ORGANIZED_DIR
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    print("\n📦 Creating DataLoaders...")
    
    # Create datasets with appropriate transforms
    train_dataset = FruitSugarDataset(
        root_dir=os.path.join(organized_dir, "train"),
        transform=get_transforms(is_training=True)
    )
    
    val_dataset = FruitSugarDataset(
        root_dir=os.path.join(organized_dir, "val"),
        transform=get_transforms(is_training=False)
    )
    
    test_dataset = FruitSugarDataset(
        root_dir=os.path.join(organized_dir, "test"),
        transform=get_transforms(is_training=False)
    )
    
    # --- Weighted Random Sampler (handles class imbalance) ---
    # If one class has fewer images, this ensures each class appears
    # equally often during training
    train_sampler = None
    shuffle_train = True
    
    if use_weighted_sampler and len(train_dataset) > 0:
        # Count samples per class
        labels = [sample[1] for sample in train_dataset.samples]
        class_counts = Counter(labels)
        
        # Calculate weight for each sample (inverse of class frequency)
        total = len(labels)
        class_weights = {cls: total / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in labels]
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle_train = False  # Sampler handles shuffling
        print("  ⚖️  Using weighted sampler to balance classes")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True  # Speeds up CPU→GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"\n  ✅ DataLoaders ready:")
    print(f"     Train:      {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"     Validation: {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"     Test:       {len(test_dataset)} images, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader
