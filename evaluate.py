"""
============================================================================
Evaluation Script — Fruit Sugar Content Estimation
============================================================================
Evaluates the trained model on the test set and generates:
  - Classification report (precision, recall, F1-score)
  - Confusion matrix visualization
  - Per-class accuracy breakdown
  - Sample predictions with confidence scores

Usage:
  python evaluate.py
============================================================================
"""

import os
import sys
import numpy as np

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SUGAR_CLASSES, RESULTS_DIR, MODEL_DIR, BEST_MODEL_FILENAME
from utils import set_seed, get_device, ensure_dir, print_separator
from dataset_utils import get_dataloaders
from model import load_model


def evaluate_model(model, test_loader, device):
    """
    Run full evaluation on the test set.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test DataLoader
        device: torch.device
    
    Returns:
        dict: Evaluation results containing predictions, labels, and metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Get probabilities with softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    results = {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    
    return results


def plot_confusion_matrix(labels, predictions, save_path=None):
    """
    Create and save a confusion matrix heatmap.
    
    The confusion matrix shows:
    - Diagonal: correctly classified samples (we want these to be high)
    - Off-diagonal: misclassified samples (we want these to be low)
    """
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SUGAR_CLASSES,
                yticklabels=SUGAR_CLASSES,
                ax=ax, square=True,
                linewidths=0.5, linecolor='gray')
    
    ax.set_title('Confusion Matrix — Sugar Content Estimation',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Class', fontsize=11)
    ax.set_ylabel('True Class', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {save_path}")
    
    plt.close(fig)


def plot_per_class_metrics(labels, predictions, save_path=None):
    """
    Create a bar chart showing precision, recall, and F1 per class.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, labels=range(len(SUGAR_CLASSES))
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(SUGAR_CLASSES))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision, width, label='Precision',
                   color='#667eea', alpha=0.85)
    bars2 = ax.bar(x, recall, width, label='Recall',
                   color='#4ECDC4', alpha=0.85)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score',
                   color='#FF6B6B', alpha=0.85)
    
    ax.set_xlabel('Sugar Content Class', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Per-Class Performance Metrics', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in SUGAR_CLASSES])
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Per-class metrics saved to: {save_path}")
    
    plt.close(fig)


def print_evaluation_report(results):
    """Print a comprehensive evaluation report to console."""
    print_separator("EVALUATION RESULTS")
    
    print(f"\n  Overall Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"  Weighted Precision: {results['precision']*100:.2f}%")
    print(f"  Weighted Recall:    {results['recall']*100:.2f}%")
    print(f"  Weighted F1-Score:  {results['f1_score']*100:.2f}%")
    
    print(f"\n{'─'*60}")
    print("\n  Detailed Classification Report:\n")
    
    report = classification_report(
        results['labels'], results['predictions'],
        target_names=SUGAR_CLASSES, digits=4
    )
    print(report)


def run_evaluation():
    """Complete evaluation pipeline."""
    print_separator("MODEL EVALUATION — Sugar Content Estimation")
    
    set_seed(42)
    device = get_device()
    
    # Load trained model
    model_path = os.path.join(MODEL_DIR, BEST_MODEL_FILENAME)
    if not os.path.exists(model_path):
        print(f"❌ No trained model found at: {model_path}")
        print("   Run 'python train.py' first to train the model.")
        return None
    
    model = load_model(model_path, device)
    
    # Load test data
    _, _, test_loader = get_dataloaders()
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    # Print report
    print_evaluation_report(results)
    
    # Save visualizations
    ensure_dir(RESULTS_DIR)
    
    plot_confusion_matrix(
        results['labels'], results['predictions'],
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png")
    )
    
    plot_per_class_metrics(
        results['labels'], results['predictions'],
        save_path=os.path.join(RESULTS_DIR, "per_class_metrics.png")
    )
    
    # Save metrics to text file
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("Fruit Sugar Content Estimation — Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy:  {results['accuracy']*100:.2f}%\n")
        f.write(f"Precision: {results['precision']*100:.2f}%\n")
        f.write(f"Recall:    {results['recall']*100:.2f}%\n")
        f.write(f"F1-Score:  {results['f1_score']*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(
            results['labels'], results['predictions'],
            target_names=SUGAR_CLASSES, digits=4
        ))
    print(f"📄 Metrics saved to: {metrics_path}")
    
    return results


if __name__ == "__main__":
    results = run_evaluation()
    if results:
        print("\n🚀 Next step: Run the demo with 'streamlit run app.py'")
