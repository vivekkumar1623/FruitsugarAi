"""
============================================================================
Model Definition — Fruit Sugar Content Estimation
============================================================================
MobileNetV2 with Transfer Learning for sugar content classification.
============================================================================
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    NUM_CLASSES, PRETRAINED, FREEZE_BACKBONE,
    UNFREEZE_LAST_N, DROPOUT_RATE, MODEL_DIR, BEST_MODEL_FILENAME
)
from utils import ensure_dir


class FruitSugarModel(nn.Module):
    """MobileNetV2-based model for fruit sugar content estimation."""
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained=PRETRAINED,
                 freeze_backbone=FREEZE_BACKBONE, dropout_rate=DROPOUT_RATE):
        super(FruitSugarModel, self).__init__()
        
        # Load Pre-trained MobileNetV2
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            self.backbone = models.mobilenet_v2(weights=weights)
            print("[OK] Loaded MobileNetV2 with ImageNet pre-trained weights")
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        # Freeze Backbone Layers (transfer learning)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze last N layers for fine-tuning
            features = list(self.backbone.features.children())
            for layer in features[-UNFREEZE_LAST_N:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"[FROZEN] Backbone frozen (except last {UNFREEZE_LAST_N} layers)")
        
        # Get feature dimensions (MobileNetV2 outputs 1280 features)
        in_features = self.backbone.classifier[1].in_features
        
        # Custom Classifier Head: 1280 → 512 → 128 → 3
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        print(f"[MODEL] Custom classifier head: {in_features} -> 512 -> 128 -> {num_classes}")
    
    def forward(self, x):
        """Forward pass: image tensor → class logits."""
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract feature vector before classifier (for visualization)."""
        x = self.backbone.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x


def get_model(num_classes=NUM_CLASSES, pretrained=PRETRAINED):
    """Create and return a FruitSugarModel instance with parameter stats."""
    model = FruitSugarModel(num_classes=num_classes, pretrained=pretrained,
                            freeze_backbone=FREEZE_BACKBONE, dropout_rate=DROPOUT_RATE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] Model Parameters:")
    print(f"   Total:     {total_params:,}")
    print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"   Frozen:    {total_params - trainable_params:,}")
    return model


def save_model(model, filepath=None):
    """Save model weights to disk."""
    if filepath is None:
        ensure_dir(MODEL_DIR)
        filepath = os.path.join(MODEL_DIR, BEST_MODEL_FILENAME)
    torch.save(model.state_dict(), filepath)
    print(f"[SAVED] Model saved to: {filepath}")


def load_model(filepath=None, device=None):
    """Load a trained model from disk."""
    if filepath is None:
        filepath = os.path.join(MODEL_DIR, BEST_MODEL_FILENAME)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FruitSugarModel(pretrained=False, freeze_backbone=False)
    state_dict = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[OK] Model loaded from: {filepath}")
    return model
