"""
============================================================================
Streamlit Web App — Fruit Sugar Content Estimation
============================================================================
Beautiful, interactive web demo for the sugar content estimation model.

Run with:
  streamlit run app.py

Features:
  - Upload a fruit image or use sample images
  - Get sugar content prediction with confidence scores
  - View visual feature analysis (color histograms, texture features)
  - Batch processing mode
============================================================================
"""

import os
import sys
import io
import tempfile

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SUGAR_CLASSES, SUGAR_DESCRIPTIONS, BRIX_RANGES,
    SELECTED_FRUITS, MODEL_DIR, BEST_MODEL_FILENAME
)
from model import load_model
from inference import predict_image, identify_fruit
from feature_extraction import extract_all_features, visualize_features


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FruitSugar AI — Sugar Content Estimator",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium Dark Theme
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .hero-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .hero-header p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Result Cards */
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 14px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .sugar-high {
        background: linear-gradient(135deg, #a8e6cf 0%, #88d8a8 100%);
        border-left: 5px solid #27ae60;
    }
    .sugar-medium {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-left: 5px solid #f39c12;
    }
    .sugar-low {
        background: linear-gradient(135deg, #fecfef 0%, #ff9a9e 100%);
        border-left: 5px solid #e74c3c;
    }
    
    /* Metric Box */
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        margin: 0.5rem 0;
    }
    .metric-box h3 {
        font-size: 0.85rem;
        color: #7f8c8d;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-box p {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0.3rem 0 0;
        color: #2c3e50;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #95a5a6;
        font-size: 0.85rem;
        margin-top: 3rem;
        border-top: 1px solid #ecf0f1;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING (cached for speed)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_trained_model():
    """Load the trained model (cached so it only loads once)."""
    model_path = os.path.join(MODEL_DIR, BEST_MODEL_FILENAME)
    if not os.path.exists(model_path):
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    return model


@st.cache_resource
def load_fruit_detector():
    """Load standard MobileNetV2 to detect if image is actually a fruit (ImageNet classes)."""
    import torchvision.models as models
    detector = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    detector.eval()
    return detector

def is_fruit(image_pil, detector):
    """Check if the image contains a fruit using ImageNet classes."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_t = transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        out = detector(img_t)
        # Get top 5 predictions
        _, indices = torch.topk(out, 5)
    
    # ImageNet indices for fruits (948 to 957 covers apples, bananas, oranges, pomegranates, lemons, etc.)
    valid_fruits = set(range(948, 958))
    
    # Check if any of the top 5 predictions are fruits
    for idx in indices[0].tolist():
        if idx in valid_fruits:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def get_sugar_class_css(class_name):
    """Return CSS class for result card based on sugar level."""
    if "High" in class_name:
        return "sugar-high"
    elif "Medium" in class_name:
        return "sugar-medium"
    return "sugar-low"


def get_sugar_emoji(class_name):
    """Return emoji for sugar level."""
    if "High" in class_name:
        return "🟢"
    elif "Medium" in class_name:
        return "🟡"
    return "🔴"


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # ── Hero Header ──
    st.markdown("""
    <div class="hero-header">
        <h1>🍎 FruitSugar AI</h1>
        <p>Non-Destructive Estimation of Sugar Content in Fruits Using Visible-Light Imaging</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 🔧 Settings")
        st.markdown("---")
        
        st.markdown("### 📋 About")
        st.markdown("""
        This system estimates the sugar content of fruits 
        using **computer vision** and **deep learning**.
        
        **How it works:**
        1. Upload a fruit photo
        2. AI analyzes color & texture patterns
        3. Get sugar content estimation instantly
        
        **Supported Fruits:**
        - 🍎 Apple
        - 🍌 Banana
        - 🍊 Orange
        - 🍈 Pomegranate
        """)
        
        st.markdown("---")
        show_features = st.checkbox("Show Feature Analysis", value=True,
                                    help="Display color histograms and texture features")
        
        st.markdown("---")
        st.markdown("### 🔬 Model Info")
        st.markdown("""
        - **Architecture:** MobileNetV2
        - **Training:** Transfer Learning
        - **Input:** 224×224 RGB images
        - **Output:** 3-class sugar estimation
        """)
    
    # ── Load Model ──
    model = load_trained_model()
    
    if model is None:
        st.error("⚠️ No trained model found! Please run `python train.py` first.")
        st.info("The model file should be at: `saved_models/best_sugar_estimator.pth`")
        return
    
    # ── Image Input ──
    st.markdown("### 📸 Provide a Fruit Image")
    
    tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Live Camera"])
    
    uploaded_file = None
    
    with tab1:
        file_upload = st.file_uploader(
            "Choose an image of a fruit (Apple, Banana, Orange, or Pomegranate)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Upload a clear photo of a single fruit"
        )
        if file_upload is not None:
            uploaded_file = file_upload
            
    with tab2:
        st.info("💡 **Viva Demo Tip:** Hold a fruit up to your webcam to test the model's accuracy on messy real-world data!")
        camera_upload = st.camera_input("Take a picture of a fruit")
        if camera_upload is not None:
            uploaded_file = camera_upload
    
    detector = load_fruit_detector()
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            # Check if it's actually a fruit
            pil_image = Image.open(tmp_path).convert("RGB")
            if not is_fruit(pil_image, detector):
                st.error("🚫 **Wait a minute!** This doesn't look like a fruit!")
                st.warning("Please upload a clear image of an Apple, Banana, Orange, or Pomegranate to estimate sugar content.")
                return
                
            # Display image and prediction side by side
            col1, col2 = st.columns([1, 1.2])
            
            with col1:
                st.markdown("#### 📷 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                
                # Detect fruit type
                fruit_name = identify_fruit(uploaded_file.name)
                if fruit_name != "unknown":
                    st.success(f"🔍 Detected fruit: **{fruit_name.capitalize()}**")
                else:
                    st.info("💡 Fruit type will be determined by the model")
            
            with col2:
                st.markdown("#### 🎯 Prediction Results")
                
                # Run prediction
                with st.spinner("🔄 Analyzing fruit..."):
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    result = predict_image(model, tmp_path, device)
                
                # Main result card
                css_class = get_sugar_class_css(result['class_name'])
                emoji = get_sugar_emoji(result['class_name'])
                
                st.markdown(f"""
                <div class="result-card {css_class}">
                    <h2 style="margin:0 0 0.5rem">{emoji} {result['class_name'].replace('_', ' ')}</h2>
                    <p style="margin:0; font-size:1rem">{result['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metric boxes
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>Confidence</h3>
                        <p>{result['confidence']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>Fruit Type</h3>
                        <p>{result['fruit_name'].capitalize()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>Est. Brix</h3>
                        <p>{result['brix_range']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.markdown("#### 📊 Probability Breakdown")
                for cls_name, prob in result['probabilities'].items():
                    display_name = cls_name.replace('_', ' ')
                    color = "#27ae60" if "High" in cls_name else (
                            "#f39c12" if "Medium" in cls_name else "#e74c3c")
                    st.markdown(f"**{display_name}**")
                    st.progress(prob / 100)
            
            # ── Feature Analysis ──
            if show_features:
                st.markdown("---")
                st.markdown("### 🔬 Visual Feature Analysis")
                st.markdown("*These features show what the model 'sees' in the image.*")
                
                with st.spinner("Extracting visual features..."):
                    image_bgr = cv2.imread(tmp_path)
                    if image_bgr is not None:
                        features = extract_all_features(image_bgr)
                        fig = visualize_features(image_bgr, features)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning("Could not load image for feature extraction.")
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    else:
        # Show placeholder when no image is uploaded
        st.markdown("""
        <div class="result-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #7f8c8d;">👆 Upload a fruit image to get started</h3>
            <p style="color: #95a5a6;">
                Supported: Apple, Banana, Orange, Pomegranate<br>
                Formats: JPG, PNG, BMP, WebP
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ── How It Works Section ──
    st.markdown("---")
    st.markdown("### 💡 How It Works")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("#### 1️⃣ Capture")
        st.markdown("Take a photo of the fruit under normal lighting conditions.")
    with c2:
        st.markdown("#### 2️⃣ Analyze")
        st.markdown("AI extracts color, texture, and surface features from the image.")
    with c3:
        st.markdown("#### 3️⃣ Predict")
        st.markdown("MobileNetV2 deep learning model estimates sugar content level.")
    with c4:
        st.markdown("#### 4️⃣ Report")
        st.markdown("Get instant results with confidence scores and Brix estimates.")
    
    # ── Footer ──
    st.markdown("""
    <div class="footer">
        <p>🧪 Fruit Sugar Content Estimation System | Computer Vision Project<br>
        Built with PyTorch, OpenCV, and Streamlit | MobileNetV2 Transfer Learning</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
