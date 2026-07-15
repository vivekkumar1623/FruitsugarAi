"""
============================================================================
Dataset Setup Script — Fruit Sugar Content Estimation
============================================================================
This script helps you download and organize the FruitNet dataset.

USAGE:
  1. Download the dataset from Kaggle:
     https://www.kaggle.com/datasets/shashwatwork/fruitnet-indian-fruits-dataset-with-quality
  
  2. Extract the ZIP file into the 'dataset/' folder in this project.
  
  3. Run this script:
     python setup_dataset.py
  
  The script will:
  - Scan for images and identify fruit types + quality labels
  - Filter for selected fruits (apple, banana, orange, pomegranate)
  - Reorganize into train/val/test splits
  - Map quality labels to sugar content classes
============================================================================
"""

import os
import sys
import shutil
import random
from collections import defaultdict

# Add project root to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_DIR, ORGANIZED_DIR, SELECTED_FRUITS,
    QUALITY_TO_SUGAR, SUGAR_CLASSES, IMAGE_EXTENSIONS,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
)
from utils import set_seed, ensure_dir, print_separator


def scan_dataset(dataset_root):
    """
    Scan the dataset directory tree and find all fruit images.
    
    This function is FLEXIBLE — it handles multiple directory structures
    by looking for quality keywords (good/bad/mixed) and fruit names
    in the file paths.
    
    Args:
        dataset_root (str): Path to the downloaded dataset folder
    
    Returns:
        list: List of (filepath, sugar_class, fruit_name) tuples
    """
    print(f"🔍 Scanning dataset directory: {dataset_root}")
    
    # Collect all samples: (filepath, sugar_class, fruit_name)
    samples = []
    skipped = 0
    
    for root, dirs, files in os.walk(dataset_root):
        for filename in files:
            # Check if file is an image
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            
            filepath = os.path.join(root, filename)
            # Convert path to lowercase for keyword matching
            path_lower = filepath.lower().replace("\\", "/")
            
            # --- Detect QUALITY from path ---
            quality = None
            for keyword, sugar_class in QUALITY_TO_SUGAR.items():
                if keyword in path_lower:
                    quality = sugar_class
                    break
            
            # --- Detect FRUIT TYPE from path ---
            fruit = None
            for fruit_name in SELECTED_FRUITS:
                if fruit_name in path_lower:
                    fruit = fruit_name
                    break
            
            # Only keep if both quality and fruit were detected
            if quality and fruit:
                samples.append((filepath, quality, fruit))
            else:
                skipped += 1
    
    print(f"  ✅ Found {len(samples)} valid images")
    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} images (unrecognized fruit/quality or other fruits)")
    
    return samples


def organize_dataset(samples, output_dir):
    """
    Organize scanned samples into train/val/test directory structure.
    
    Creates the following structure:
    organized_data/
    ├── train/
    │   ├── Low_Sugar/
    │   ├── Medium_Sugar/
    │   └── High_Sugar/
    ├── val/
    │   ├── Low_Sugar/
    │   ├── Medium_Sugar/
    │   └── High_Sugar/
    └── test/
        ├── Low_Sugar/
        ├── Medium_Sugar/
        └── High_Sugar/
    
    Args:
        samples (list): List of (filepath, sugar_class, fruit_name) tuples
        output_dir (str): Root directory for organized data
    """
    print_separator("Organizing Dataset")
    
    # Set seed for reproducible splits
    set_seed(42)
    
    # Group samples by class for stratified splitting
    class_samples = defaultdict(list)
    for filepath, sugar_class, fruit in samples:
        class_samples[sugar_class].append((filepath, fruit))
    
    # Print class distribution
    print("\n📊 Class distribution (before split):")
    for cls in SUGAR_CLASSES:
        count = len(class_samples[cls])
        print(f"  {cls}: {count} images")
    
    # Create directory structure
    splits = ["train", "val", "test"]
    for split in splits:
        for cls in SUGAR_CLASSES:
            ensure_dir(os.path.join(output_dir, split, cls))
    
    # Split each class independently (stratified split)
    split_counts = {"train": 0, "val": 0, "test": 0}
    
    for cls in SUGAR_CLASSES:
        items = class_samples[cls]
        random.shuffle(items)  # Shuffle for random split
        
        n_total = len(items)
        n_train = int(n_total * TRAIN_SPLIT)
        n_val = int(n_total * VAL_SPLIT)
        # Remaining goes to test
        
        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]
        
        # Copy files to organized directories
        for split_name, split_items in [("train", train_items), 
                                          ("val", val_items), 
                                          ("test", test_items)]:
            for filepath, fruit in split_items:
                # Create unique filename: fruit_originalname.ext
                original_name = os.path.basename(filepath)
                new_name = f"{fruit}_{original_name}"
                dest = os.path.join(output_dir, split_name, cls, new_name)
                
                # Avoid overwriting if names collide
                counter = 1
                while os.path.exists(dest):
                    name, ext = os.path.splitext(new_name)
                    dest = os.path.join(output_dir, split_name, cls, f"{name}_{counter}{ext}")
                    counter += 1
                
                shutil.copy2(filepath, dest)
                split_counts[split_name] += 1
    
    # Print final summary
    print_separator("Organization Complete")
    total = sum(split_counts.values())
    for split in splits:
        pct = split_counts[split] / total * 100 if total > 0 else 0
        print(f"  {split.capitalize():10s}: {split_counts[split]:5d} images ({pct:.1f}%)")
    print(f"  {'Total':10s}: {total:5d} images")
    print(f"\n📁 Organized data saved to: {output_dir}")


def validate_organized_dataset(organized_dir):
    """Check that the organized dataset looks correct."""
    print_separator("Validation Check")
    
    all_good = True
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(organized_dir, split)
        if not os.path.exists(split_dir):
            print(f"  ❌ Missing: {split_dir}")
            all_good = False
            continue
        
        for cls in SUGAR_CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.exists(cls_dir):
                print(f"  ❌ Missing: {cls_dir}")
                all_good = False
                continue
            
            count = len([f for f in os.listdir(cls_dir) 
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])
            status = "✅" if count > 0 else "⚠️ "
            print(f"  {status} {split}/{cls}: {count} images")
    
    if all_good:
        print("\n🎉 Dataset is ready for training!")
    else:
        print("\n⚠️  Some issues detected — check the paths above.")
    
    return all_good


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print_separator("Fruit Sugar Content — Dataset Setup")
    
    # Step 1: Check if dataset directory exists
    if not os.path.exists(DATASET_DIR):
        print(f"\n❌ Dataset directory not found: {DATASET_DIR}")
        print(f"\n📥 Please follow these steps:")
        print(f"   1. Go to: https://www.kaggle.com/datasets/shashwatwork/fruitnet-indian-fruits-dataset-with-quality")
        print(f"   2. Click 'Download' (you need a Kaggle account)")
        print(f"   3. Extract the ZIP into: {DATASET_DIR}")
        print(f"   4. Run this script again: python setup_dataset.py")
        sys.exit(1)
    
    # Step 2: Scan for images
    samples = scan_dataset(DATASET_DIR)
    
    if len(samples) == 0:
        print("\n❌ No valid images found!")
        print("   Make sure the dataset is extracted correctly.")
        print(f"   Expected structure inside {DATASET_DIR}:")
        print(f"     ├── Good Quality/  (or similar)")
        print(f"     │   ├── apple/")
        print(f"     │   ├── banana/")
        print(f"     │   └── ...")
        print(f"     ├── Bad Quality/")
        print(f"     └── Mixed Quality/")
        sys.exit(1)
    
    # Step 3: Check if organized data already exists
    if os.path.exists(ORGANIZED_DIR):
        print(f"\n⚠️  Organized data already exists at: {ORGANIZED_DIR}")
        response = input("   Delete and re-organize? (y/N): ").strip().lower()
        if response == 'y':
            shutil.rmtree(ORGANIZED_DIR)
            print("   Deleted existing organized data.")
        else:
            print("   Keeping existing data. Running validation only...")
            validate_organized_dataset(ORGANIZED_DIR)
            sys.exit(0)
    
    # Step 4: Organize into train/val/test
    organize_dataset(samples, ORGANIZED_DIR)
    
    # Step 5: Validate
    validate_organized_dataset(ORGANIZED_DIR)
    
    print("\n🚀 Next step: Run training with 'python train.py'")
