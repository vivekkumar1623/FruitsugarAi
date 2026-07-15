# 🍎 FruitSugar AI

Non-destructive fruit sugar content estimation using deep learning and computer vision.

## 🔬 About

FruitSugar AI uses **MobileNetV2 transfer learning** to estimate sugar content levels (High / Medium / Low) in fruits by analyzing their visual appearance — color, texture, and ripeness indicators.

- **Accuracy:** 99.68% on 14,700+ images
- **Fruits Supported:** Apple, Banana, Orange, Pomegranate
- **Deployment:** Real-time Streamlit web app with file upload and live camera support

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow, PyTorch, MobileNetV2
- **Computer Vision:** OpenCV, PIL
- **ML Tools:** scikit-learn, NumPy, Matplotlib
- **Web App:** Streamlit
- **Dataset:** FruitNet (Indian Fruits Dataset with Quality)

## 📦 Dataset

This project uses the **FruitNet: Indian Fruits Dataset with Quality** from Kaggle.

🔗 **Download:** [https://www.kaggle.com/datasets/shashwatwork/fruitnet-indian-fruits-dataset-with-quality](https://www.kaggle.com/datasets/shashwatwork/fruitnet-indian-fruits-dataset-with-quality)

After downloading, extract the ZIP into the `dataset/` folder and run:
```bash
python setup_dataset.py
```

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/vivekkumar1623/FruitsugarAi.git
cd FruitsugarAi
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download & setup dataset
Download from the [Kaggle link above](#-dataset), extract into `dataset/`, then:
```bash
python setup_dataset.py
```

### 4. Train the model
```bash
python train.py
```

### 5. Run the app
```bash
streamlit run app.py
```

## 📁 Project Structure

```
FruitsugarAi/
├── app.py                 # Streamlit web application
├── model.py               # MobileNetV2 model architecture
├── train.py               # Training script
├── evaluate.py            # Model evaluation & metrics
├── inference.py           # Prediction on new images
├── config.py              # Configuration settings
├── dataset_utils.py       # Data loading & transforms
├── feature_extraction.py  # Feature extraction utilities
├── setup_dataset.py       # Dataset organization script
├── download_dataset.py    # Kaggle dataset downloader
├── utils.py               # Helper functions
├── requirements.txt       # Python dependencies
└── .gitignore             # Git ignore rules
```

## 📊 Results

| Metric | Value |
|--------|-------|
| Accuracy | 99.68% |
| Model | MobileNetV2 (Transfer Learning) |
| Images | 14,700+ |
| Classes | High Sugar, Medium Sugar, Low Sugar |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Vivek Kumar** — [GitHub](https://github.com/vivekkumar1623)
