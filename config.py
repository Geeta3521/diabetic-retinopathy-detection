"""
config.py — Central configuration for Eye Disease Detection System
All hyperparameters, paths, and class definitions live here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data", "dataset")
TRAIN_DIR       = DATA_DIR
TEST_DIR        = DATA_DIR
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "eye_disease_model.h5")
MODEL_WEIGHTS   = os.path.join(BASE_DIR, "models", "eye_disease_weights.weights.h5")
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")
ASSETS_DIR      = os.path.join(BASE_DIR, "assets")

# ─── Classes ──────────────────────────────────────────────────────────────────
CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

CLASS_INFO = {
    "normal": {
        "display_name": "Normal",
        "description": "No signs of eye disease detected.",
        "recommendation": "Maintain regular eye check-ups every 1-2 years.",
        "severity": "None",
        "color": "#22c55e"   # green
    },
    "cataract": {
        "display_name": "Cataract",
        "description": "Clouding of the eye's natural lens detected.",
        "recommendation": "Consult an ophthalmologist for cataract evaluation. Surgical correction is highly effective.",
        "severity": "Moderate",
        "color": "#f59e0b"   # amber
    },
    "diabetic_retinopathy": {
        "display_name": "Diabetic Retinopathy",
        "description": "Damage to retinal blood vessels, linked to diabetes.",
        "recommendation": "Urgent: Consult an ophthalmologist and endocrinologist. Blood sugar control is critical.",
        "severity": "High",
        "color": "#ef4444"   # red
    },
    "glaucoma": {
        "display_name": "Glaucoma",
        "description": "Elevated intraocular pressure causing optic nerve damage.",
        "recommendation": "Immediate consultation with an ophthalmologist. Early treatment prevents vision loss.",
        "severity": "High",
        "color": "#8b5cf6"   # purple
    }
}

# ─── Image Settings ───────────────────────────────────────────────────────────
IMAGE_SIZE      = (224, 224)
CHANNELS        = 3
INPUT_SHAPE     = (224, 224, 3)

# ─── Training Hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS_FROZEN   = 3     # Phase 1: train head only (frozen base)
EPOCHS_FINETUNE = 3   # Phase 2: fine-tune top layers
LEARNING_RATE   = 1e-3
FINETUNE_LR     = 1e-5
VALIDATION_SPLIT = 0.2
RANDOM_SEED     = 42

# ─── EfficientNetB0 Fine-tuning ───────────────────────────────────────────────
UNFREEZE_FROM_LAYER = 100    # unfreeze layers from this index onwards

# ─── Grad-CAM ─────────────────────────────────────────────────────────────────
GRADCAM_LAYER   = "top_conv"   # last conv layer in EfficientNetB0
GRADCAM_ALPHA   = 0.5          # overlay transparency

# ─── Report ───────────────────────────────────────────────────────────────────
HOSPITAL_NAME   = "AI Eye Care Diagnostics"
REPORT_AUTHOR   = "AI-Based Eye Disease Detection System v1.0"
