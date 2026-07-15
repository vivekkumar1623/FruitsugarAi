"""
============================================================================
Feature Extraction — Fruit Sugar Content Estimation
============================================================================
Extracts visual features from fruit images for analysis & explainability.

Features extracted:
  1. Color Histograms (HSV & LAB color spaces)
  2. GLCM Texture Features (contrast, energy, homogeneity, correlation)
  3. Local Binary Patterns (LBP) for surface texture
  
These features help explain WHY the model makes its predictions.
============================================================================
"""

import os
import sys
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR
from utils import ensure_dir


def extract_color_histogram(image_bgr, bins=64):
    """
    Extract color histograms in HSV and LAB color spaces.
    
    Why HSV and LAB?
    - HSV separates Hue (color), Saturation (intensity), Value (brightness)
    - LAB separates Lightness from color (a=green-red, b=blue-yellow)
    - Both are better than RGB for analyzing fruit ripeness/sugar content
    
    Args:
        image_bgr (np.array): Input image in BGR format (OpenCV default)
        bins (int): Number of histogram bins
    
    Returns:
        dict: Histograms for each channel
    """
    # Convert BGR → HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Convert BGR → LAB (Lightness, A-channel, B-channel)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    
    histograms = {}
    
    # HSV histograms
    for i, name in enumerate(["Hue", "Saturation", "Value"]):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms[f"HSV_{name}"] = hist
    
    # LAB histograms
    for i, name in enumerate(["Lightness", "A_channel", "B_channel"]):
        hist = cv2.calcHist([lab], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms[f"LAB_{name}"] = hist
    
    return histograms


def extract_glcm_features(image_bgr):
    """
    Extract GLCM (Gray-Level Co-occurrence Matrix) texture features.
    
    GLCM analyzes spatial relationships between pixel intensities.
    These features capture surface texture patterns that correlate
    with fruit ripeness and sugar content.
    
    Features:
      - Contrast: intensity difference between neighboring pixels
      - Energy: uniformity of texture (smooth vs rough)
      - Homogeneity: closeness of pixel values to diagonal
      - Correlation: linear dependency of gray levels
    
    Args:
        image_bgr (np.array): Input image in BGR format
    
    Returns:
        dict: GLCM feature values
    """
    # Convert to grayscale for texture analysis
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Resize to consistent size for fair comparison
    gray = cv2.resize(gray, (128, 128))
    
    # Reduce gray levels for GLCM computation (256 → 32 levels)
    gray_reduced = (gray // 8).astype(np.uint8)
    
    # Compute GLCM at multiple angles for rotation invariance
    # distances=[1]: immediate neighbor; angles: 0°, 45°, 90°, 135°
    glcm = graycomatrix(gray_reduced, distances=[1], 
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=32, symmetric=True, normed=True)
    
    # Extract properties (average across all angles)
    features = {
        "GLCM_Contrast":    float(np.mean(graycoprops(glcm, 'contrast'))),
        "GLCM_Energy":      float(np.mean(graycoprops(glcm, 'energy'))),
        "GLCM_Homogeneity": float(np.mean(graycoprops(glcm, 'homogeneity'))),
        "GLCM_Correlation": float(np.mean(graycoprops(glcm, 'correlation'))),
    }
    
    return features


def extract_lbp_features(image_bgr, radius=2, n_points=16):
    """
    Extract Local Binary Pattern (LBP) features.
    
    LBP captures micro-texture patterns by comparing each pixel
    to its neighbors. Different patterns indicate different surface
    textures (smooth, rough, spotted, etc.).
    
    Args:
        image_bgr (np.array): Input image in BGR format
        radius (int): Radius of the circular LBP pattern
        n_points (int): Number of neighbor points to compare
    
    Returns:
        dict: LBP histogram and statistics
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    
    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Create histogram of LBP patterns
    n_bins = n_points + 2  # n_points uniform patterns + 1 non-uniform
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    features = {
        "LBP_Histogram": hist,
        "LBP_Mean": float(np.mean(lbp)),
        "LBP_Std": float(np.std(lbp)),
    }
    
    return features


def extract_all_features(image_bgr):
    """
    Extract all visual features from a single image.
    
    Args:
        image_bgr (np.array): Input image in BGR format
    
    Returns:
        dict: All extracted features
    """
    features = {}
    features["color"] = extract_color_histogram(image_bgr)
    features["glcm"] = extract_glcm_features(image_bgr)
    features["lbp"] = extract_lbp_features(image_bgr)
    return features


def visualize_features(image_bgr, features, save_path=None):
    """
    Create a visualization of all extracted features.
    
    Generates a multi-panel figure showing:
    - Original image
    - HSV color histograms
    - LAB color histograms
    - GLCM texture metrics (bar chart)
    - LBP histogram
    
    Args:
        image_bgr (np.array): Original image (BGR)
        features (dict): Features from extract_all_features()
        save_path (str): If provided, save figure to this path
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Visual Feature Analysis for Sugar Content Estimation",
                 fontsize=14, fontweight='bold')
    
    # Panel 1: Original Image
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original Image", fontsize=11)
    axes[0, 0].axis('off')
    
    # Panel 2: HSV Histograms
    colors_hsv = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, name in enumerate(["Hue", "Saturation", "Value"]):
        key = f"HSV_{name}"
        if key in features["color"]:
            axes[0, 1].plot(features["color"][key], color=colors_hsv[i],
                           label=name, alpha=0.8, linewidth=1.5)
    axes[0, 1].set_title("HSV Color Distribution", fontsize=11)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].set_xlabel("Bin")
    axes[0, 1].set_ylabel("Frequency")
    
    # Panel 3: LAB Histograms
    colors_lab = ['#95E1D3', '#F38181', '#FCE38A']
    for i, name in enumerate(["Lightness", "A_channel", "B_channel"]):
        key = f"LAB_{name}"
        if key in features["color"]:
            axes[0, 2].plot(features["color"][key], color=colors_lab[i],
                           label=name, alpha=0.8, linewidth=1.5)
    axes[0, 2].set_title("LAB Color Distribution", fontsize=11)
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].set_xlabel("Bin")
    axes[0, 2].set_ylabel("Frequency")
    
    # Panel 4: GLCM Texture Features (bar chart)
    glcm = features["glcm"]
    metrics = list(glcm.keys())
    values = list(glcm.values())
    bar_colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
    bars = axes[1, 0].bar(range(len(metrics)), values, color=bar_colors)
    axes[1, 0].set_xticks(range(len(metrics)))
    axes[1, 0].set_xticklabels([m.replace("GLCM_", "") for m in metrics],
                                fontsize=9, rotation=15)
    axes[1, 0].set_title("GLCM Texture Features", fontsize=11)
    axes[1, 0].set_ylabel("Value")
    # Add value labels on bars
    for bar, val in zip(bars, values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Panel 5: LBP Histogram
    lbp_hist = features["lbp"]["LBP_Histogram"]
    axes[1, 1].bar(range(len(lbp_hist)), lbp_hist, color='#a8e6cf', edgecolor='#3d8b6e')
    axes[1, 1].set_title("LBP Texture Pattern Distribution", fontsize=11)
    axes[1, 1].set_xlabel("Pattern ID")
    axes[1, 1].set_ylabel("Frequency")
    
    # Panel 6: Summary Statistics
    axes[1, 2].axis('off')
    summary_text = "Feature Summary\n" + "─" * 30 + "\n\n"
    summary_text += f"GLCM Contrast:     {glcm['GLCM_Contrast']:.4f}\n"
    summary_text += f"GLCM Energy:       {glcm['GLCM_Energy']:.4f}\n"
    summary_text += f"GLCM Homogeneity:  {glcm['GLCM_Homogeneity']:.4f}\n"
    summary_text += f"GLCM Correlation:  {glcm['GLCM_Correlation']:.4f}\n"
    summary_text += f"\nLBP Mean:          {features['lbp']['LBP_Mean']:.4f}\n"
    summary_text += f"LBP Std Dev:       {features['lbp']['LBP_Std']:.4f}\n"
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Feature visualization saved to: {save_path}")
    
    return fig
