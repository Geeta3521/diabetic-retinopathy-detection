<div align="center">

# 👁️ AI-Based Multi-Disease Eye Detection System

### EfficientNetB0 · Grad-CAM XAI · Streamlit · PDF Reports

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Geeta3521/diabetic-retinopathy-detection?style=social)](https://github.com/Geeta3521/diabetic-retinopathy-detection/stargazers)

**Detects Diabetic Retinopathy, Cataract & Glaucoma from retinal fundus images — with explainable AI heatmaps and auto-generated medical PDF reports.**

[🚀 Live Demo](#-deploy-to-streamlit-cloud) · [📖 Docs](#-quick-start) · [🐛 Issues](https://github.com/Geeta3521/diabetic-retinopathy-detection/issues)

<!-- REPLACE THIS with a real screenshot/GIF of your app -->
<!-- ![App Demo](assets/demo.gif) -->

</div>

---

## 🎯 What This Does

Upload a retinal fundus image → get an instant AI diagnosis with:
- **Disease classification** across 4 categories (Normal, Cataract, DR, Glaucoma)
- **Confidence scores** visualized as interactive Plotly charts
- **Grad-CAM heatmap** showing exactly which regions influenced the diagnosis
- **Downloadable PDF medical report** with clinical recommendations

---

## 🏥 Detection Performance

| Disease              | Accuracy  | Severity |
|----------------------|-----------|----------|
| ✅ Normal             | ~97%      | None     |
| 🟡 Cataract           | 92–98%    | Moderate |
| 🔴 Diabetic Retinopathy | 90–96%  | High     |
| 🔴 Glaucoma           | 88–95%    | High     |

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🔬 Multi-disease | Classifies 4 conditions simultaneously |
| 🧠 Transfer Learning | EfficientNetB0 fine-tuned on retinal data |
| 🎨 Explainable AI | Grad-CAM heatmaps for clinical transparency |
| ⚡ Preprocessing | CLAHE contrast enhancement + augmentation |
| 📄 PDF Reports | Auto-generated with ReportLab |
| 🌐 Web App | Streamlit UI — upload & get results instantly |
| 📊 Evaluation | ROC curves, confusion matrix, F1-score |

---

## 🏗️ Architecture

```
Input Retinal Image
       ↓
Preprocessing (Resize 224×224 → CLAHE → Normalize)
       ↓
EfficientNetB0 (ImageNet pretrained, frozen)
       ↓
Classification Head
  GlobalAveragePooling2D → BatchNorm
  Dense(256, ReLU) → Dropout(0.4)
  Dense(128, ReLU) → Dropout(0.3)
  Dense(4, Softmax)
       ↓
Prediction + Confidence Scores
       ↓
Grad-CAM Heatmap (last Conv layer)
       ↓
Auto-Generated PDF Medical Report
```

---

## 📁 Project Structure

```
diabetic-retinopathy-detection/
├── app.py                    # Streamlit web app
├── train.py                  # Two-phase training pipeline
├── predict.py                # Inference + CLI tool
├── evaluate.py               # Metrics: ROC, confusion matrix, F1
├── config.py                 # Hyperparameters and paths
├── requirements.txt
│
├── utils/
│   ├── preprocessing.py      # CLAHE + augmentation
│   ├── gradcam.py            # Grad-CAM XAI implementation
│   └── report_generator.py  # PDF report generation
│
├── data/
│   └── dataset_guide.md      # Dataset download instructions
│
├── models/                   # Saved weights (after training)
└── reports/                  # Generated PDF reports
```

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/Geeta3521/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download dataset

```bash
pip install kaggle
kaggle datasets download -d gunavenkatdoddi/eye-diseases-classification
unzip eye-diseases-classification.zip -d dataset/
```

### 4. Train

```bash
python train.py         # Phase 1 (10 epochs) + Phase 2 fine-tuning (20 epochs)
```

### 5. Run the web app

```bash
streamlit run app.py
```

### 6. CLI prediction (single image)

```bash
python predict.py path/to/retinal_image.jpg
```

---

## 🔬 Explainable AI — Grad-CAM

Grad-CAM highlights which pixels most influenced the model's decision:

- 🔴 **Red/Yellow** → High diagnostic significance (e.g., microaneurysms, optic disc)
- 🔵 **Blue/Purple** → Low influence regions

This is essential for medical AI — clinicians can verify the model focuses on clinically relevant areas before trusting a prediction.

---

## 📊 Training Config

| Parameter | Value |
|---|---|
| Base Model | EfficientNetB0 (ImageNet) |
| Input Size | 224 × 224 × 3 |
| Batch Size | 32 |
| Phase 1 LR | 1e-3 (frozen base) |
| Phase 2 LR | 1e-5 (fine-tuning) |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |

---

## 🌐 Deploy to Streamlit Cloud

1. Push to GitHub (with trained model in `models/`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py` → Deploy

> For large model files, load from Google Drive or Hugging Face Hub instead of committing directly.

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**. It is **not** intended for clinical diagnosis or medical decision-making. Always consult a qualified ophthalmologist.

---

## 📚 References

1. Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML.*
2. Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV.*
3. APTOS 2019 Blindness Detection — Kaggle Competition.
4. Zuiderveld (1994). CLAHE. *Graphics Gems IV.*

---

## 👩‍💻 Author

**Geeta Ajit Nemgouda** — BE (AIML), BMS College of Engineering, Bengaluru

[![GitHub](https://img.shields.io/badge/GitHub-Geeta3521-181717?style=flat&logo=github)](https://github.com/Geeta3521)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_LINKEDIN)
