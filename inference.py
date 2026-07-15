"""
============================================================================
Inference Module — Fruit Sugar Content Estimation
============================================================================
Predict sugar content of fruit images using the trained model.

Usage:
  # Single image
  python inference.py --image path/to/fruit.jpg
  
  # Batch (folder)
  python inference.py --folder path/to/images/
============================================================================
"""

import os
import sys
import argparse

import cv2
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SUGAR_CLASSES, SUGAR_DESCRIPTIONS, BRIX_RANGES,
    SELECTED_FRUITS, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    IMAGE_EXTENSIONS, MODEL_DIR, BEST_MODEL_FILENAME
)
from model import load_model
from feature_extraction import extract_all_features, visualize_features
from dataset_utils import get_transforms


def identify_fruit(image_path):
    """
    Simple fruit identification from filename or path.
    In a production system, this would be a separate classifier.
    
    Args:
        image_path (str): Path to the image
    
    Returns:
        str: Detected fruit name or 'unknown'
    """
    path_lower = image_path.lower()
    for fruit in SELECTED_FRUITS:
        if fruit in path_lower:
            return fruit
    return "unknown"


def predict_image(model, image_path, device=None):
    """
    Predict sugar content for a single fruit image.
    
    Args:
        model: Trained PyTorch model
        image_path (str): Path to the fruit image
        device: torch.device (auto-detected if None)
    
    Returns:
        dict: Prediction results containing:
            - class_name: Predicted sugar level (str)
            - class_index: Predicted class index (int)
            - confidence: Confidence percentage (float)
            - probabilities: Per-class probabilities (dict)
            - fruit_name: Detected fruit type (str)
            - brix_range: Estimated Brix range (str)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = get_transforms(is_training=False)
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dim
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # Apply Temperature Scaling to soften the probabilities
        # temperature=1.7 lowers confidence by ~6-7% vs raw softmax
        temperature = 1.4
        probabilities = torch.nn.functional.softmax(outputs / temperature, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, 0)
    
    # Build result
    class_idx = predicted_class.item()
    class_name = SUGAR_CLASSES[class_idx]
    fruit_name = identify_fruit(image_path)
    
    # Get estimated Brix range
    brix_range = "N/A"
    if fruit_name in BRIX_RANGES and class_name in BRIX_RANGES[fruit_name]:
        brix_range = BRIX_RANGES[fruit_name][class_name]
    
    result = {
        "class_name": class_name,
        "class_index": class_idx,
        "confidence": confidence.item() * 100,
        "probabilities": {
            SUGAR_CLASSES[i]: probabilities[i].item() * 100
            for i in range(len(SUGAR_CLASSES))
        },
        "fruit_name": fruit_name,
        "brix_range": brix_range,
        "description": SUGAR_DESCRIPTIONS.get(class_name, ""),
    }
    
    return result


def predict_with_features(model, image_path, device=None, save_dir=None):
    """
    Predict sugar content AND extract visual features for explainability.
    
    Args:
        model: Trained model
        image_path (str): Path to fruit image
        device: torch.device
        save_dir (str): Directory to save feature visualization
    
    Returns:
        tuple: (prediction_dict, features_dict, feature_figure)
    """
    # Get prediction
    prediction = predict_image(model, image_path, device)
    
    # Extract visual features
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"❌ Could not read image: {image_path}")
        return prediction, None, None
    
    features = extract_all_features(image_bgr)
    
    # Visualize features
    save_path = None
    if save_dir:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_dir, f"features_{basename}.png")
    
    fig = visualize_features(image_bgr, features, save_path=save_path)
    
    return prediction, features, fig


def predict_batch(model, folder_path, device=None):
    """
    Predict sugar content for all images in a folder.
    
    Args:
        model: Trained model
        folder_path (str): Path to folder containing fruit images
        device: torch.device
    
    Returns:
        list: List of (filename, prediction_dict) tuples
    """
    results = []
    
    for filename in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        
        filepath = os.path.join(folder_path, filename)
        try:
            prediction = predict_image(model, filepath, device)
            results.append((filename, prediction))
            
            print(f"  {filename}: {prediction['class_name']} "
                  f"({prediction['confidence']:.1f}% confidence)")
        except Exception as e:
            print(f"  ❌ {filename}: Error — {e}")
    
    return results


def print_prediction(result, image_path=""):
    """Print a nicely formatted prediction result."""
    print(f"\n{'═'*55}")
    print(f"  🍎 Sugar Content Estimation Result")
    print(f"{'═'*55}")
    if image_path:
        print(f"  Image:      {os.path.basename(image_path)}")
    print(f"  Fruit:      {result['fruit_name'].capitalize()}")
    print(f"  Sugar Level: {result['class_name'].replace('_', ' ')}")
    print(f"  Confidence:  {result['confidence']:.1f}%")
    print(f"  Brix Range:  {result['brix_range']}")
    print(f"  {result['description']}")
    print(f"\n  Probability Breakdown:")
    for cls, prob in result['probabilities'].items():
        bar = '█' * int(prob / 5) + '░' * (20 - int(prob / 5))
        print(f"    {cls.replace('_', ' '):14s} |{bar}| {prob:.1f}%")
    print(f"{'═'*55}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN (CLI)
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict sugar content of fruit images"
    )
    parser.add_argument("--image", type=str, help="Path to a single fruit image")
    parser.add_argument("--folder", type=str, help="Path to folder of fruit images")
    parser.add_argument("--features", action="store_true",
                        help="Also extract and visualize visual features")
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        parser.print_help()
        sys.exit(1)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device=device)
    
    if args.image:
        if args.features:
            result, features, fig = predict_with_features(
                model, args.image, device, save_dir="results"
            )
        else:
            result = predict_image(model, args.image, device)
        print_prediction(result, args.image)
    
    elif args.folder:
        print(f"\n📁 Batch prediction for: {args.folder}\n")
        results = predict_batch(model, args.folder, device)
        print(f"\n✅ Processed {len(results)} images")
