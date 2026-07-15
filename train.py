"""
============================================================================
Training Script — Fruit Sugar Content Estimation
============================================================================
Trains the MobileNetV2 model on the organized FruitNet dataset.

Usage:
  python train.py

What it does:
  1. Loads the organized dataset (train/val splits)
  2. Creates a MobileNetV2 model with transfer learning
  3. Trains with early stopping and learning rate scheduling
  4. Saves the best model based on validation accuracy
  5. Generates training curves (loss & accuracy plots)
============================================================================
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE,
    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR,
    MODEL_DIR, RESULTS_DIR, BEST_MODEL_FILENAME
)
from utils import set_seed, get_device, ensure_dir, format_time, print_separator
from dataset_utils import get_dataloaders
from model import get_model, save_model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one complete epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: torch.device
    
    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    model.train()  # Enable training mode (dropout, batch norm active)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to GPU/CPU
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass: input → model → predictions
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update weights
        
        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)  # Get class with highest score
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Evaluate model on validation set (no gradient computation).
    
    Args:
        model: PyTorch model
        val_loader: Validation DataLoader
        criterion: Loss function
        device: torch.device
    
    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    model.eval()  # Disable dropout, batch norm uses running stats
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No gradients needed for validation
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def plot_training_curves(history, save_path=None):
    """
    Plot training & validation loss/accuracy curves.
    
    These curves help diagnose:
    - Overfitting (val loss increases while train loss decreases)
    - Underfitting (both losses remain high)
    - Good fit (both losses decrease together)
    
    Args:
        history (dict): Training history with loss/accuracy per epoch
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Val Loss', markersize=4)
    ax1.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-o', label='Val Acc', markersize=4)
    ax2.set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Training curves saved to: {save_path}")
    
    plt.close(fig)


def train_model():
    """
    Complete training pipeline with early stopping.
    
    Returns:
        tuple: (trained_model, training_history)
    """
    print_separator("TRAINING — Fruit Sugar Content Estimation")
    
    # Step 1: Setup
    set_seed(42)
    device = get_device()
    
    # Step 2: Load data
    train_loader, val_loader, _ = get_dataloaders()
    
    # Step 3: Create model
    model = get_model()
    model = model.to(device)
    
    # Step 4: Define loss function and optimizer
    # CrossEntropyLoss: standard for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer: adaptive learning rate, works well out-of-the-box
    # Only optimize parameters that require gradients (unfrozen layers)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler: reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_SCHEDULER_PATIENCE,
        factor=LR_SCHEDULER_FACTOR
    )
    
    # Step 5: Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    
    print_separator("Starting Training")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"  Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
              f"LR: {current_lr:.6f} | Time: {format_time(epoch_time)}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model)
            patience_counter = 0
            print(f"  🏆 New best validation accuracy: {best_val_acc:.1f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping triggered after {epoch} epochs "
                  f"(no improvement for {PATIENCE} epochs)")
            break
    
    total_time = time.time() - start_time
    print_separator("Training Complete")
    print(f"  Total training time: {format_time(total_time)}")
    print(f"  Best validation accuracy: {best_val_acc:.1f}%")
    print(f"  Model saved to: {os.path.join(MODEL_DIR, BEST_MODEL_FILENAME)}")
    
    # Plot training curves
    plot_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plot_training_curves(history, save_path=plot_path)
    
    return model, history


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model, history = train_model()
    print("\n🚀 Next step: Run evaluation with 'python evaluate.py'")
