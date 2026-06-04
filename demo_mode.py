"""
demo_mode.py — Demo mode for testing the app WITHOUT a trained model.

This creates a mock model that returns random-ish predictions so you can
test the full Streamlit UI flow before your real model is trained.

Usage:
    python demo_mode.py       # Creates a lightweight demo model
    streamlit run app.py      # Then run the app normally
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import INPUT_SHAPE, MODEL_SAVE_PATH, CLASS_NAMES


def build_demo_model(num_classes: int = 4) -> Model:
    """
    Lightweight demo model (no EfficientNet) — instant to create,
    gives plausible-looking outputs for UI testing.
    NOT suitable for real medical use.
    """
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="input_image")
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = Model(inputs=inputs, outputs=outputs, name="DemoEyeModel")
    return model


def create_demo_model():
    print("\n" + "="*55)
    print("  DEMO MODE — Creating lightweight test model")
    print("="*55)
    print("\n  ⚠️  This is a DEMO model with random weights.")
    print("  It will produce predictions but NOT medically accurate.")
    print("  Run train.py with real data to get the actual model.\n")

    os.makedirs("models", exist_ok=True)

    model = build_demo_model(num_classes=4)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Save
    model.save(MODEL_SAVE_PATH)

    # Save class indices (alphabetical order)
    class_indices = {name: i for i, name in enumerate(sorted(CLASS_NAMES))}
    with open("models/class_indices.json", "w") as f:
        json.dump(class_indices, f, indent=2)

    print(f"  Demo model saved to: {MODEL_SAVE_PATH}")
    print(f"  Class indices: {class_indices}")
    print("\n  You can now run: streamlit run app.py")
    print("  (The app will use this demo model for UI testing)\n")


if __name__ == "__main__":
    create_demo_model()
