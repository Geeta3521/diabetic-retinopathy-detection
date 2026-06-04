import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import TRAIN_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED

def get_data_generators():
    """
    Returns (train_gen, val_gen, test_gen, class_indices)
    Auto-splits single flat dataset folder into train/val/test.
    """
    import math

    # Step 1 — scan all image paths and labels
    all_paths, all_labels, class_names = [], [], []
    for class_name in sorted(os.listdir(TRAIN_DIR)):
        class_path = os.path.join(TRAIN_DIR, class_name)
        if not os.path.isdir(class_path):            continue
        class_names.append(class_name)
        for fname in os.listdir(class_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                all_paths.append(os.path.join(class_path, fname))
                all_labels.append(class_name)

    import random
    random.seed(RANDOM_SEED)
    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)

    n = len(all_paths)
    n_train = int(n * 0.80)
    n_val   = int(n * 0.10)

    train_paths  = list(all_paths[:n_train])
    train_labels = list(all_labels[:n_train])
    val_paths    = list(all_paths[n_train:n_train + n_val])
    val_labels   = list(all_labels[n_train:n_train + n_val])
    test_paths   = list(all_paths[n_train + n_val:])
    test_labels  = list(all_labels[n_train + n_val:])

    class_indices = {name: i for i, name in enumerate(sorted(class_names))}
    num_classes   = len(class_names)

    print(f"  Total images : {n}")
    print(f"  Train        : {len(train_paths)}")
    print(f"  Validation   : {len(val_paths)}")
    print(f"  Test         : {len(test_paths)}")
    print(f"  Classes      : {class_indices}")

    def make_generator(paths, labels, augment=False):
        while True:
            indices = list(range(len(paths)))
            if augment:
                random.shuffle(indices)
            for start in range(0, len(indices), BATCH_SIZE):
                batch_idx = indices[start:start + BATCH_SIZE]
                imgs, labs = [], []
                for i in batch_idx:
                    img = cv2.imread(paths[i])
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, IMAGE_SIZE)
                    img = apply_clahe(img)
                    if augment:
                        # Random horizontal flip
                        if random.random() > 0.5:
                            img = cv2.flip(img, 1)
                        # Random brightness
                        factor = random.uniform(0.85, 1.15)
                        img = np.clip(img * factor, 0, 255).astype(np.uint8)
                        # Random rotation ±15°
                        angle = random.uniform(-15, 15)
                        M = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
                        img = cv2.warpAffine(img, M, IMAGE_SIZE)
                    img = img.astype(np.float32) / 255.0
                    label_vec = np.zeros(num_classes, dtype=np.float32)
                    label_vec[class_indices[labels[i]]] = 1.0
                    imgs.append(img)
                    labs.append(label_vec)
                if imgs:
                    yield np.array(imgs), np.array(labs)

    # Wrap as objects with .samples and .classes attributes for compatibility
    from tensorflow.keras.utils import Sequence

    class GenWrapper(Sequence):
        def __init__(self, paths, labels, augment):
            self.paths = paths
            self.labels = labels
            self.augment = augment
            self.samples = len(paths)
            self.classes = np.array([class_indices[l] for l in labels])

        def __len__(self):
            return int(np.ceil(self.samples / BATCH_SIZE))

        def __getitem__(self, idx):
            import cv2

            start = idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, self.samples)

            imgs = []
            labs = []

            for i in range(start, end):
                img = cv2.imread(self.paths[i])

                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMAGE_SIZE)

                img = img.astype(np.float32) / 255.0

                label_vec = np.zeros(num_classes, dtype=np.float32)
                label_vec[class_indices[self.labels[i]]] = 1.0

                imgs.append(img)
                labs.append(label_vec)

            return np.array(imgs), np.array(labs)

        def on_epoch_end(self):
            pass

        def reset(self):
            pass


    train_gen = GenWrapper(
        train_paths,
        train_labels,
        augment=True
        )

    val_gen = GenWrapper(
        val_paths,
        val_labels,
        augment=False
        )

    test_gen = GenWrapper(
        test_paths,
        test_labels,
        augment=False
        )

    return train_gen, val_gen, test_gen, class_indices
import cv2
import numpy as np
from PIL import Image

def preprocess_pil_image(pil_img):
    """
    Preprocess uploaded PIL image for prediction.
    """

    img = np.array(pil_img)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, IMAGE_SIZE)

    img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)

    return img


def preprocess_image_from_path(image_path):
    """
    Preprocess image from file path.
    """

    img = Image.open(image_path).convert("RGB")

    return preprocess_pil_image(img)
def preprocess_for_display(img):
    """
    Helper function for Grad-CAM visualization.
    """

    import numpy as np

    if isinstance(img, np.ndarray):
        display_img = img.copy()

        if display_img.max() <= 1.0:
            display_img = (display_img * 255).astype(np.uint8)

        return display_img

    return img