"""
Download the FruitNet dataset using kagglehub.
Run this FIRST before setup_dataset.py

Usage:
  pip install kagglehub
  python download_dataset.py
"""

import os
import shutil

# Step 1: Install kagglehub if not already installed
try:
    import kagglehub
except ImportError:
    print("Installing kagglehub...")
    os.system("pip install kagglehub")
    import kagglehub

# Step 2: Download dataset
print("📥 Downloading FruitNet dataset from Kaggle...")
print("   (This is ~3 GB, may take a few minutes)\n")

path = kagglehub.dataset_download("shashwatwork/fruitnet-indian-fruits-dataset-with-quality")

print(f"\n✅ Downloaded to: {path}")

# Step 3: Copy to project's dataset/ folder
project_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(project_dir, "dataset")

if os.path.exists(dataset_dir):
    print(f"\n⚠️  'dataset/' folder already exists.")
    print(f"   Kaggle data is at: {path}")
    print(f"   You can use it directly or copy manually.")
else:
    print(f"\n📁 Copying to project's dataset/ folder...")
    shutil.copytree(path, dataset_dir)
    print(f"✅ Dataset copied to: {dataset_dir}")

print(f"\n🚀 Next step: python setup_dataset.py")
