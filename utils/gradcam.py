"""
utils/gradcam.py — Grad-CAM Explainability
Generates heatmaps showing which regions influenced the model's prediction.
"""

import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGE_SIZE, GRADCAM_ALPHA


def find_last_conv_layer(model) -> str:
    """Auto-detect the last convolutional layer name in the model."""
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
        # Also handle EfficientNet's internal conv layers
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    last_conv = sublayer.name
    return last_conv


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    pred_index: int = None
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap.

    Args:
        img_array: Preprocessed image (1, 224, 224, 3) float32
        model: Trained Keras model
        pred_index: Class index to explain. If None, uses argmax of prediction.

    Returns:
        heatmap: np.ndarray (224, 224) float32 in [0, 1]
    """
    # Build a sub-model: inputs → (last_conv_output, final_predictions)
    # Find last conv layer inside the EfficientNet base
    base_model = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is not None:
        # Find last conv inside EfficientNet
        last_conv_layer = None
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer

        if last_conv_layer is None:
            # Fallback: use top_activation or similar
            for layer in base_model.layers:
                if "activation" in layer.name.lower() and "top" in layer.name.lower():
                    last_conv_layer = layer
                    break

        if last_conv_layer is None:
            raise ValueError("Could not find a suitable conv layer for Grad-CAM in EfficientNet.")

        # Build grad model from base input
        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, base_model.output]
        )

        # Wrap to include classification head
        with tf.GradientTape() as tape:
            base_outputs = grad_model(img_array, training=False)
            conv_outputs, base_features = base_outputs

            # Pass through classification head
            x = base_features
            for layer in model.layers:
                if "efficientnet" in layer.name.lower():
                    continue
                x = layer(x)
            predictions = x

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

    else:
        # Simple model (no nested base_model) — fallback approach
        last_conv_name = None
        for layer in model.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D,)):
                last_conv_name = layer.name

        if last_conv_name is None:
            raise ValueError("No Conv2D layer found in model.")

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

    # Pool gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    return heatmap


def overlay_gradcam(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = GRADCAM_ALPHA,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        original_img: uint8 RGB image (H, W, 3)
        heatmap: float32 heatmap (any size), will be resized
        alpha: heatmap transparency
        colormap: OpenCV colormap

    Returns:
        superimposed: uint8 RGB image (H, W, 3)
    """
    h, w = original_img.shape[:2]

    # Resize heatmap to image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to uint8 and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend with original
    superimposed = cv2.addWeighted(original_img, 1 - alpha, heatmap_rgb, alpha, 0)
    return superimposed


def generate_gradcam_figure(
    original_pil: Image.Image,
    preprocessed_array: np.ndarray,
    model: tf.keras.Model,
    predicted_class: str,
    confidence: float,
    pred_index: int
) -> plt.Figure:
    """
    Generate a 3-panel matplotlib figure:
    [Original | CLAHE Preprocessed | Grad-CAM Overlay]

    Returns:
        matplotlib Figure object
    """
    from utils.preprocessing import preprocess_for_display

    orig_np = np.array(original_pil.convert("RGB"))
    orig_resized = cv2.resize(orig_np, IMAGE_SIZE)
    preprocessed_display = preprocess_for_display(original_pil)

    try:
        heatmap = make_gradcam_heatmap(preprocessed_array, model, pred_index)
        overlay = overlay_gradcam(orig_resized, heatmap)
    except Exception as e:
        print(f"[Grad-CAM Warning] {e} — showing blank heatmap")
        overlay = orig_resized.copy()
        heatmap = np.zeros((7, 7))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0f172a")

    titles = ["Original Image", "CLAHE Enhanced", f"Grad-CAM — {predicted_class}"]
    images = [orig_resized, preprocessed_display, overlay]

    for ax, title, img in zip(axes, titles, images):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Add colorbar for heatmap
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.set_label("Activation Intensity", color="white", fontsize=10)

    plt.suptitle(
        f"Prediction: {predicted_class}  |  Confidence: {confidence:.1f}%",
        color="white", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    return fig


def save_gradcam_figure(fig: plt.Figure, save_path: str) -> str:
    """Save figure to disk, return path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    print("Grad-CAM module loaded ✓")
