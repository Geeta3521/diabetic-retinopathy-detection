# Dataset Guide — Eye Disease Detection

## Datasets Required

### 1. Diabetic Retinopathy (DR)
- **Source**: APTOS 2019 Blindness Detection (Kaggle)
- **URL**: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
- **Classes used**: 0 (No DR) → label as "Normal", 1–4 → label as "Diabetic_Retinopathy"
- **Recommended images**: 1000 Normal + 1000 DR for quick training

### 2. Cataract
- **Source**: Ocular Disease Intelligent Recognition (ODIR-5K)
- **URL**: https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
- **OR**: Eye Diseases Classification dataset
- **URL**: https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification
- **Classes used**: cataract folder

### 3. Glaucoma
- **Source**: RIM-ONE / REFUGE / ORIGA
- **URL (easiest)**: https://www.kaggle.com/datasets/sshikamaru/glaucoma-detection
- **OR from ODIR above**: glaucoma folder

---

## Recommended Quick Setup (All-in-One Dataset)

Use this single Kaggle dataset that already has all 4 classes separated:
**Eye Diseases Classification** — https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification

```
dataset/
├── train/
│   ├── normal/        (~1074 images)
│   ├── diabetic_retinopathy/  (~1098 images)
│   ├── cataract/      (~1038 images)
│   └── glaucoma/      (~1007 images)
└── test/
    ├── normal/
    ├── diabetic_retinopathy/
    ├── cataract/
    └── glaucoma/
```

**Download Command (with Kaggle API):**
```bash
pip install kaggle
kaggle datasets download -d gunavenkatdoddi/eye-diseases-classification
unzip eye-diseases-classification.zip -d dataset/
```

---

## Expected Folder Structure After Download

Place inside the project root:
```
eye_disease_detection/
├── dataset/
│   ├── train/
│   │   ├── normal/
│   │   ├── diabetic_retinopathy/
│   │   ├── cataract/
│   │   └── glaucoma/
│   └── test/
│       ├── normal/
│       ├── diabetic_retinopathy/
│       ├── cataract/
│       └── glaucoma/
```

---

## Notes
- Minimum ~800 images per class for decent accuracy
- CLAHE preprocessing handles uneven lighting across datasets
- If mixing datasets, ensure class balance with augmentation
