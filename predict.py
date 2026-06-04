"""
predict.py — Inference module for Eye Disease Detection
Load trained model and run predictions with confidence scores.
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_SAVE_PATH, CLASS_NAMES, CLASS_INFO
from utils.preprocessing import preprocess_pil_image, preprocess_image_from_path


# ─── Model Loader ─────────────────────────────────────────────────────────────

_model_cache = None
_class_indices_cache = None


def load_model(model_path: str = MODEL_SAVE_PATH) -> tf.keras.Model:
    """Load and cache the trained Keras model."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Please run train.py first to train the model."
        )

    print(f"[Predictor] Loading model from {model_path}...")
    _model_cache = tf.keras.models.load_model(model_path)
    print("[Predictor] Model loaded ✓")
    return _model_cache


def load_class_indices(path: str = "models/class_indices.json") -> dict:
    """Load class indices mapping {class_name: index}."""
    global _class_indices_cache
    if _class_indices_cache is not None:
        return _class_indices_cache

    if os.path.exists(path):
        with open(path) as f:
            _class_indices_cache = json.load(f)
    else:
        # Default ordering (alphabetical — matches Keras default)
        _class_indices_cache = {name: i for i, name in enumerate(sorted(CLASS_NAMES))}

    return _class_indices_cache


def get_idx_to_class(class_indices: dict) -> dict:
    """Invert class_indices to {index: class_name}."""
    return {v: k for k, v in class_indices.items()}


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict_from_pil(pil_image: Image.Image) -> dict:
    """
    Run full prediction on a PIL Image.

    Returns:
        {
            "predicted_class": str,
            "display_name": str,
            "confidence": float,          # 0–100
            "probabilities": {cls: prob}, # all classes
            "class_info": dict,
            "preprocessed_array": np.ndarray  # for Grad-CAM
        }
    """
    model = load_model()
    class_indices = load_class_indices()
    idx_to_class = get_idx_to_class(class_indices)

    # Preprocess
    img_array = preprocess_pil_image(pil_image)  # (1, 224, 224, 3)

    # Inference
   # preds = model.predict(img_array, verbose=0)[0]  # (num_classes,)
    print("Shape:", img_array.shape)
    print("Min:", img_array.min())
    print("Max:", img_array.max())

    preds = model.predict(img_array, verbose=0)[0]

    print("Raw prediction:", preds)
    # Results
    pred_idx = int(np.argmax(preds))
    predicted_class = idx_to_class[pred_idx]
    confidence = float(preds[pred_idx]) * 100.0

    probabilities = {
        idx_to_class[i]: float(preds[i]) * 100.0
        for i in range(len(preds))
    }

    return {
        "predicted_class": predicted_class,
        "display_name": CLASS_INFO.get(predicted_class, {}).get("display_name", predicted_class),
        "confidence": confidence,
        "probabilities": probabilities,
        "pred_index": pred_idx,
        "class_info": CLASS_INFO.get(predicted_class, {}),
        "preprocessed_array": img_array
    }


def predict_from_path(image_path: str) -> dict:
    """Run prediction on an image file path."""
    img = Image.open(image_path).convert("RGB")
    return predict_from_pil(img)


def batch_predict(image_paths: list) -> list:
    """Run predictions on a list of image file paths."""
    model = load_model()
    class_indices = load_class_indices()
    idx_to_class = get_idx_to_class(class_indices)

    results = []
    for path in image_paths:
        try:
            result = predict_from_path(path)
            result["image_path"] = path
            results.append(result)
        except Exception as e:
            results.append({"image_path": path, "error": str(e)})

    return results


# ─── CLI Usage ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eye Disease Prediction CLI")
    parser.add_argument("image_path", type=str, help="Path to eye image")
    args = parser.parse_args()

    result = predict_from_path(args.image_path)
    print("\n" + "="*50)
    print("  EYE DISEASE DETECTION RESULT")
    print("="*50)
    print(f"  Condition  : {result['display_name']}")
    print(f"  Confidence : {result['confidence']:.2f}%")
    print(f"  Severity   : {result['class_info'].get('severity', 'N/A')}")
    print(f"\n  All Probabilities:")
    for cls, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob / 5)
        print(f"    {cls:<25} {prob:5.1f}%  {bar}")
    print(f"\n  Recommendation:")
    print(f"    {result['class_info'].get('recommendation', 'N/A')}")
    print("="*50 + "\n")
