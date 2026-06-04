"""
train.py — EfficientNetB0 Transfer Learning + Fine-tuning
Run this script to train the eye disease detection model.

Usage:
    python train.py

Requirements:
    - dataset/ folder set up (see data/dataset_guide.md)
    - All packages from requirements.txt installed
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from config import (
    INPUT_SHAPE, CLASS_NAMES, BATCH_SIZE,
    EPOCHS_FROZEN, EPOCHS_FINETUNE,
    LEARNING_RATE, FINETUNE_LR,
    MODEL_SAVE_PATH, MODEL_WEIGHTS,
    UNFREEZE_FROM_LAYER, RANDOM_SEED
)
from utils.preprocessing import get_data_generators

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─── Model Architecture ───────────────────────────────────────────────────────

def build_model(num_classes: int = 4, trainable_base: bool = False) -> Model:
    """
    Build EfficientNetB0-based classifier.

    Architecture:
        EfficientNetB0 (ImageNet weights, frozen)
        → GlobalAveragePooling2D
        → BatchNorm → Dense(256, relu) → Dropout(0.4)
        → BatchNorm → Dense(128, relu) → Dropout(0.3)
        → Dense(num_classes, softmax)
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE
    )
    base_model.trainable = trainable_base

    inputs = tf.keras.Input(shape=INPUT_SHAPE)

    # EfficientNetB0 expects uint8 or [0,255] input — we've already normalized
    # so set include_preprocessing=False behaviour via direct call
    x = base_model(inputs)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="EyeDiseaseDetector")
    return model


def unfreeze_top_layers(model: Model, from_layer: int = UNFREEZE_FROM_LAYER):
    """Unfreeze EfficientNet layers from `from_layer` onwards for fine-tuning."""
    base_model = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        print("[Warning] EfficientNet base not found for unfreezing.")
        return

    base_model.trainable = True
    for layer in base_model.layers[:from_layer]:
        layer.trainable = False

    print(f"  Unfroze layers {from_layer}+ of EfficientNetB0 ({len(base_model.layers)} total layers)")


# ─── Training ─────────────────────────────────────────────────────────────────

def compute_class_weights(train_gen) -> dict:
    """Compute class weights to handle dataset imbalance."""
    from sklearn.utils.class_weight import compute_class_weight
    labels = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {i: w for i, w in zip(classes, weights)}


def plot_training_history(history_phase1, history_phase2, save_path: str):
    """Plot and save training curves for both phases."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f172a")

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    # Combine histories
    acc  = history_phase1.history["accuracy"] + history_phase2.history["accuracy"]
    val_acc = history_phase1.history["val_accuracy"] + history_phase2.history["val_accuracy"]
    loss = history_phase1.history["loss"] + history_phase2.history["loss"]
    val_loss = history_phase1.history["val_loss"] + history_phase2.history["val_loss"]
    phase_split = len(history_phase1.history["accuracy"])

    epochs = range(1, len(acc) + 1)

    axes[0].plot(epochs, acc, "#3b82f6", linewidth=2, label="Train")
    axes[0].plot(epochs, val_acc, "#22c55e", linewidth=2, label="Validation")
    axes[0].axvline(phase_split, color="#f59e0b", linestyle="--", alpha=0.7, label="Fine-tuning starts")
    axes[0].set_title("Model Accuracy", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(facecolor="#1e293b", labelcolor="white")

    axes[1].plot(epochs, loss, "#ef4444", linewidth=2, label="Train")
    axes[1].plot(epochs, val_loss, "#f97316", linewidth=2, label="Validation")
    axes[1].axvline(phase_split, color="#f59e0b", linestyle="--", alpha=0.7, label="Fine-tuning starts")
    axes[1].set_title("Model Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(facecolor="#1e293b", labelcolor="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Training curves saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_xlabel("Predicted Label", color="white")
    ax.set_ylabel("True Label", color="white")
    ax.set_title("Confusion Matrix", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Confusion matrix saved to: {save_path}")


def train():
    """Full training pipeline."""
    print("\n" + "="*60)
    print("  EYE DISEASE DETECTION — TRAINING PIPELINE")
    print("="*60)

    # ── GPU check ──
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n  GPU(s) available: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # ── Data ──
    print("\n[1/5] Loading data generators...")

    train_gen, val_gen, test_gen, class_indices = get_data_generators() 
    print(f"  Class mapping  : {class_indices}")
    from sklearn.utils.class_weight import compute_class_weight
    cw_labels  = train_gen.classes
    cw_classes = np.unique(cw_labels)
    cw_values  = compute_class_weight("balanced", classes=cw_classes, y=cw_labels)
    class_weights = {int(i): float(w) for i, w in zip(cw_classes, cw_values)}
    print(f"  Class weights  : {class_weights}")



    num_classes = len(class_indices)
    os.makedirs("models", exist_ok=True)

    # ── Phase 1: Frozen base ──
    print("\n[2/5] Building model (frozen EfficientNetB0)...")
    model = build_model(num_classes=num_classes, trainable_base=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    model.summary()

    callbacks_phase1 = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy", verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]

    print(f"\n[3/5] Phase 1 Training — {EPOCHS_FROZEN} epochs (frozen base)...")
    history_p1 = model.fit(
        train_gen,
        epochs=EPOCHS_FROZEN,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_phase1,
        verbose=1
    )

    # ── Phase 2: Fine-tuning ──
    print(f"\n[4/5] Phase 2 Fine-tuning — {EPOCHS_FINETUNE} epochs (top layers unfrozen)...")
    unfreeze_top_layers(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )

    callbacks_phase2 = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=7, restore_best_weights=True, monitor="val_accuracy", verbose=1),
        ReduceLROnPlateau(factor=0.3, patience=4, min_lr=1e-8, verbose=1),
    ]

    history_p2 = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_phase2,
        verbose=1
)
# ── Evaluation ──
    print("\n[5/5] Evaluating on test set...")

    results = model.evaluate(test_gen, verbose=1)

    metric_names = [
       "loss",
       "accuracy",
       "auc",
       "precision",
       "recall"
    ]

    for name, val in zip(metric_names, results):
      print(f"  Test {name}: {val:.4f}")

    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true = test_gen.classes 

    idx_to_class = {v: k for k, v in class_indices.items()}
    target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names
        )
    )

    plot_training_history(history_p1, history_p2, "models/training_curves.png")
    plot_confusion_matrix(y_true, y_pred, target_names, "models/confusion_matrix.png")

    # Save class indices
    with open("models/class_indices.json", "w") as f:
        json.dump(class_indices, f, indent=2)

    print(f"\n  Model saved to : {MODEL_SAVE_PATH}")
    print("  Training complete ✓\n")
    return model


if __name__ == "__main__":
    train()
