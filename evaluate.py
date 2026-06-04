"""
evaluate.py — Model Evaluation & Metrics
Run after training to generate full evaluation report.

Usage:
    python evaluate.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_SAVE_PATH, CLASS_NAMES
from utils.preprocessing import get_data_generators

DARK_BG     = "#0f172a"
CARD_BG     = "#1e293b"
ACCENT      = "#3b82f6"
COLORS      = ["#3b82f6", "#22c55e", "#ef4444", "#8b5cf6"]


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Normalized)"],
        ["d", ".2f"]
    ):
        ax.set_facecolor(CARD_BG)
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.5, linecolor="#334155",
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_xlabel("Predicted", color="white", fontsize=12)
        ax.set_ylabel("True", color="white", fontsize=12)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
        ax.tick_params(colors="white", labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def plot_roc_curves(y_true, y_pred_probs, class_names, save_path):
    """Plot one-vs-rest ROC curves for all classes."""
    from sklearn.preprocessing import label_binarize
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    for i, (cls, color) in enumerate(zip(class_names, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
    ax.set_ylabel("True Positive Rate", color="white", fontsize=12)
    ax.set_title("ROC Curves — Multi-class (One vs Rest)", color="white",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.15, color="white")
    legend = ax.legend(facecolor=CARD_BG, labelcolor="white", fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ROC curves saved: {save_path}")


def plot_per_class_metrics(report_dict, class_names, save_path):
    """Bar chart of precision, recall, F1 per class."""
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    for i, (metric, color) in enumerate(zip(metrics, ["#3b82f6", "#22c55e", "#f59e0b"])):
        vals = [report_dict[cls][metric] for cls in class_names]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                      color=color, alpha=0.85, edgecolor="none")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", color="white",
                    fontsize=9, fontweight="bold")

    ax.set_xlabel("Disease Class", color="white", fontsize=12)
    ax.set_ylabel("Score", color="white", fontsize=12)
    ax.set_title("Per-Class Precision / Recall / F1-Score", color="white",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.set_ylim([0, 1.12])
    ax.grid(axis="y", alpha=0.15, color="white")
    legend = ax.legend(facecolor=CARD_BG, labelcolor="white", fontsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Per-class metrics saved: {save_path}")


def evaluate():
    print("\n" + "="*60)
    print("  EYE DISEASE DETECTION — EVALUATION")
    print("="*60)

    # Load model
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"[ERROR] Model not found at {MODEL_SAVE_PATH}. Run train.py first.")
        sys.exit(1)

    print("\n[1/3] Loading model...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    model.summary()

    # Load data
    print("\n[2/3] Loading test data...")
    _, _, test_gen, class_indices = get_data_generators()
    idx_to_class = {v: k for k, v in class_indices.items()}
    ordered_classes = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    # Predict
    test_gen.reset()
    print("  Running inference on test set...")
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    # Test metrics
    print("\n[3/3] Computing metrics...")
    results = model.evaluate(test_gen, verbose=0)
    metric_names = model.metrics_names
    print("\n  Test Metrics:")
    for name, val in zip(metric_names, results):
        print(f"    {name:<20}: {val:.4f}")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=ordered_classes)
    report_dict = classification_report(
        y_true, y_pred, target_names=ordered_classes, output_dict=True
    )
    print("\n  Classification Report:")
    print(report)

    # Save text report
    os.makedirs("models", exist_ok=True)
    with open("models/evaluation_report.txt", "w") as f:
        f.write("EYE DISEASE DETECTION — EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        for name, val in zip(metric_names, results):
            f.write(f"{name}: {val:.4f}\n")
        f.write("\n" + report)

    # Plots
    plot_confusion_matrix(y_true, y_pred, ordered_classes, "models/confusion_matrix.png")
    plot_roc_curves(y_true, y_pred_probs, ordered_classes, "models/roc_curves.png")
    plot_per_class_metrics(report_dict, ordered_classes, "models/per_class_metrics.png")

    print("\n  All evaluation artifacts saved to models/")
    print("  Evaluation complete ✓\n")


if __name__ == "__main__":
    evaluate()
