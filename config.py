"""
============================================================================
Configuration File — Fruit Sugar Content Estimation
============================================================================
All hyperparameters, paths, and constants are defined here.
Change values here instead of modifying other files.
============================================================================
"""

import os

# ═══════════════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Dataset location — points to where kagglehub downloaded the data
DATASET_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache", "kagglehub", "datasets", "shashwatwork",
    "fruitnet-indian-fruits-dataset-with-quality", "versions", "1"
)
ORGANIZED_DIR = os.path.join(BASE_DIR, "organized_data")  # Reorganized train/val/test
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")        # Saved model weights
RESULTS_DIR = os.path.join(BASE_DIR, "results")           # Plots, reports, outputs

# ═══════════════════════════════════════════════════════════════════════════
# DATASET CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
# We focus on 4 fruits for a clean demo
SELECTED_FRUITS = ["apple", "banana", "orange", "pomegranate"]

# Mapping: dataset quality labels → sugar content estimation
# Good quality = ripe, optimal sugar → High Sugar
# Mixed quality = partially ripe       → Medium Sugar
# Bad quality  = unripe/overripe       → Low Sugar
QUALITY_TO_SUGAR = {
    "good":  "High_Sugar",
    "bad":   "Low_Sugar",
    "mixed": "Medium_Sugar",
    "fresh": "High_Sugar",    # Alternative naming in some datasets
    "rotten": "Low_Sugar",    # Alternative naming in some datasets
}

# Class names in order (index 0, 1, 2)
SUGAR_CLASSES = ["Low_Sugar", "Medium_Sugar", "High_Sugar"]
NUM_CLASSES = len(SUGAR_CLASSES)

# Sugar content descriptions for UI display
SUGAR_DESCRIPTIONS = {
    "High_Sugar":   "🟢 High Sugar Content — Fruit is optimally ripe with peak sweetness.",
    "Medium_Sugar": "🟡 Medium Sugar Content — Fruit is partially ripe, moderate sweetness.",
    "Low_Sugar":    "🔴 Low Sugar Content — Fruit is unripe or overripe, low sweetness.",
}

# Estimated Brix ranges (for display purposes — based on food science literature)
BRIX_RANGES = {
    "apple":       {"High_Sugar": "13–16°Bx", "Medium_Sugar": "10–13°Bx", "Low_Sugar": "6–10°Bx"},
    "banana":      {"High_Sugar": "20–25°Bx", "Medium_Sugar": "14–20°Bx", "Low_Sugar": "5–14°Bx"},
    "orange":      {"High_Sugar": "11–15°Bx", "Medium_Sugar": "8–11°Bx",  "Low_Sugar": "5–8°Bx"},
    "pomegranate": {"High_Sugar": "15–19°Bx", "Medium_Sugar": "12–15°Bx", "Low_Sugar": "8–12°Bx"},
}

# ═══════════════════════════════════════════════════════════════════════════
# IMAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
IMG_SIZE = 224           # MobileNetV2 expects 224×224 input
IMG_CHANNELS = 3         # RGB images
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ═══════════════════════════════════════════════════════════════════════════
# DATA SPLIT RATIOS
# ═══════════════════════════════════════════════════════════════════════════
TRAIN_SPLIT = 0.70       # 70% for training
VAL_SPLIT = 0.15         # 15% for validation
TEST_SPLIT = 0.15        # 15% for testing

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
BATCH_SIZE = 32          # Images per batch (reduce to 16 if memory issues)
NUM_WORKERS = 2          # DataLoader worker threads
EPOCHS = 20              # Maximum training epochs
LEARNING_RATE = 0.001    # Initial learning rate (Adam optimizer)
WEIGHT_DECAY = 1e-4      # L2 regularization to prevent overfitting
PATIENCE = 5             # Early stopping patience (epochs without improvement)
LR_SCHEDULER_PATIENCE = 3  # Reduce LR after N epochs without improvement
LR_SCHEDULER_FACTOR = 0.5  # Multiply LR by this factor when reducing

# ═══════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
MODEL_NAME = "mobilenet_v2"   # Backbone architecture
PRETRAINED = True             # Use ImageNet pre-trained weights
FREEZE_BACKBONE = True        # Freeze early layers (transfer learning)
UNFREEZE_LAST_N = 4           # Fine-tune last N layers of the backbone
DROPOUT_RATE = 0.3            # Dropout for regularization in classifier head
BEST_MODEL_FILENAME = "best_sugar_estimator.pth"

# ═══════════════════════════════════════════════════════════════════════════
# AUGMENTATION SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
# Training augmentations for robustness
AUGMENTATION = {
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.2,
    "rotation_degrees": 15,
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
    "color_jitter_saturation": 0.2,
    "color_jitter_hue": 0.1,
}

# ImageNet normalization values (required for pre-trained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
