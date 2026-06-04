# 👁 AI-Based Multi-Disease Eye Disease Detection System

> **Deep Learning + Explainable AI for Diabetic Retinopathy, Cataract & Glaucoma Detection**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Project Overview

A production-ready AI system that detects **three major eye diseases** from retinal fundus images using **EfficientNetB0 Transfer Learning** and **Grad-CAM Explainable AI**. Includes a Streamlit web app and auto-generated PDF medical reports.

### Diseases Detected
| Disease | Expected Accuracy | Severity |
|---------|------------------|----------|
| Normal | ~97% | None |
| Cataract | 92–98% | Moderate |
| Diabetic Retinopathy | 90–96% | High |
| Glaucoma | 88–95% | High |

---

## ✨ Key Features

- **Multi-disease classification** — 4-class: Normal, Cataract, DR, Glaucoma
- **EfficientNetB0 Transfer Learning** — ImageNet pretrained + fine-tuned
- **CLAHE Preprocessing** — Contrast-limited adaptive histogram equalization
- **Grad-CAM XAI** — Visual heatmaps explaining model decisions
- **PDF Report Generation** — Professional medical report with ReportLab
- **Streamlit Web App** — Upload image → instant result → download report
- **Confidence Breakdown** — Probability scores for all classes (Plotly charts)

---

## 🏗️ Architecture

```
Input Image (any size)
       ↓
Preprocessing
  ├── Resize to 224×224
  ├── CLAHE Enhancement (LAB color space)
  └── Normalize [0, 1]
       ↓
EfficientNetB0 (ImageNet weights, frozen)
       ↓
Classification Head
  ├── GlobalAveragePooling2D
  ├── BatchNorm → Dense(256, ReLU) → Dropout(0.4)
  ├── BatchNorm → Dense(128, ReLU) → Dropout(0.3)
  └── Dense(4, Softmax)
       ↓
Prediction + Confidence Scores
       ↓
Grad-CAM Heatmap (last Conv layer)
       ↓
PDF Medical Report
```

---

## 📁 Project Structure

```
eye_disease_detection/
├── app.py                    # Streamlit web application
├── train.py                  # Model training (Phase 1 + Phase 2)
├── predict.py                # Inference module + CLI
├── evaluate.py               # Evaluation metrics + plots
├── config.py                 # All hyperparameters and paths
├── requirements.txt          # Python dependencies
│
├── utils/
│   ├── preprocessing.py      # CLAHE + augmentation pipeline
│   ├── gradcam.py            # Grad-CAM XAI implementation
│   └── report_generator.py  # PDF report with ReportLab
│
├── data/
│   └── dataset_guide.md      # Dataset download instructions
│
├── models/                   # Saved model files (after training)
│   ├── eye_disease_model.h5
│   ├── class_indices.json
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── roc_curves.png
│
├── reports/                  # Generated PDF reports
└── .streamlit/
    └── config.toml           # Streamlit theme config
```

---

## 🚀 Quick Start

### 1. Clone / Download the Project
```bash
git clone https://github.com/Geeta3521/diabetic-retinopathy-detection.gitcd eye_disease_detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Follow instructions in `data/dataset_guide.md`.

**Easiest option** — single Kaggle dataset with all 4 classes:
```bash
pip install kaggle
kaggle datasets download -d gunavenkatdoddi/eye-diseases-classification
unzip eye-diseases-classification.zip -d dataset/
```

Ensure structure:
```
dataset/
├── train/
│   ├── normal/
│   ├── diabetic_retinopathy/
│   ├── cataract/
│   └── glaucoma/
└── test/
    ├── normal/
    ├── diabetic_retinopathy/
    ├── cataract/
    └── glaucoma/
```

### 4. Train the Model
```bash
python train.py
```
This runs:
- **Phase 1** (10 epochs): Train classification head, frozen EfficientNetB0
- **Phase 2** (20 epochs): Fine-tune top layers with low LR

### 5. Evaluate
```bash
python evaluate.py
```
Generates confusion matrix, ROC curves, per-class metrics.

### 6. Run Web App
```bash
streamlit run app.py
```

### 7. CLI Prediction (single image)
```bash
python predict.py path/to/eye_image.jpg
```

---

## 🌐 Deploy to Streamlit Cloud

1. Push project to GitHub (include trained model in `models/`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo → select `app.py` → Deploy

> **Note**: Model file (`eye_disease_model.h5`) must be in the repo or loaded from Google Drive/Hugging Face Hub for cloud deployment.

### Loading model from Google Drive (for cloud deployment):
```python
# In predict.py, replace load_model() with:
import gdown
gdown.download("YOUR_GDRIVE_LINK", "models/eye_disease_model.h5", quiet=False)
```

---

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | EfficientNetB0 (ImageNet) |
| Input Size | 224 × 224 × 3 |
| Batch Size | 32 |
| Phase 1 LR | 1e-3 (frozen base) |
| Phase 2 LR | 1e-5 (fine-tuning) |
| Phase 1 Epochs | 10 |
| Phase 2 Epochs | 20 |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Augmentation | Rotation, Flip, Zoom, Brightness |

---

## 🔬 Explainable AI — Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Maps) highlights the pixels that most influenced the model's decision:

- **Red/Yellow regions** → High diagnostic significance
- **Blue/Purple regions** → Low influence

This is crucial for medical AI credibility — clinicians can verify the model is looking at relevant regions (e.g., optic disc for glaucoma, microaneurysms for DR).

---

## 📄 Medical Report Contents

Auto-generated PDF includes:
- Patient information (name, age, ID)
- Detected condition + confidence score
- Severity level + clinical recommendation
- Confidence breakdown (all 4 classes)
- Grad-CAM visualization (3-panel)
- Timestamp + disclaimer

---

## 📚 Research Gap (for Paper Writing)

> "Existing AI systems for ophthalmological screening predominantly address single eye diseases in isolation, limiting their clinical utility. This work proposes a unified deep learning framework leveraging EfficientNetB0 transfer learning for simultaneous multi-disease classification across Diabetic Retinopathy, Cataract, and Glaucoma, augmented with Gradient-weighted Class Activation Maps (Grad-CAM) for explainable predictions. The proposed system achieves competitive accuracy while providing visual transparency essential for clinical adoption."

---

## 🏆 Why This Project Stands Out

| Feature | This Project | Typical Student Project |
|---------|-------------|------------------------|
| Diseases | 3 diseases + Normal | Single disease |
| Architecture | EfficientNetB0 + Fine-tuning | Basic CNN |
| Explainability | Grad-CAM XAI | None |
| Preprocessing | CLAHE + Augmentation | Basic resize |
| Deployment | Streamlit Cloud | Local only |
| Reports | Auto-generated PDF | None |
| Evaluation | ROC + Conf. Matrix + F1 | Accuracy only |

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**. It is NOT intended for clinical diagnosis or medical decision-making. Always consult a qualified ophthalmologist for medical advice.

---

## 📖 References

1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML.
2. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV.
3. Acharya, U. R., et al. (2018). Automated diabetic macular edema detection using deep learning. IEEE.
4. APTOS 2019 Blindness Detection — Kaggle Competition.
5. Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. Graphics Gems IV.
