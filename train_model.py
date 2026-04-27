import os
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                         ReduceLROnPlateau)

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR    = "collected_data"   # Folder created by collect_data.py
MODEL_DIR   = "model"            # Where to save trained model & metadata
IMG_SIZE    = (64, 64)           # Input size for the CNN (smaller = faster)
BATCH_SIZE  = 32
EPOCHS      = 30
TEST_SPLIT  = 0.2
RANDOM_SEED = 42


# ─────────────────────────────────────────────
#  STEP 1 – LOAD IMAGES FROM DISK
# ─────────────────────────────────────────────
def load_dataset(data_dir):
    """
    Walk through each class sub-folder, load images, return
    (images_array, labels_list).
    """
    images, labels = [], []
    class_names = sorted(os.listdir(data_dir))

    print(f"[INFO] Found {len(class_names)} classes in '{data_dir}'")

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        class_images = 0
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(class_path, img_file)
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=IMG_SIZE
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(class_name)
            class_images += 1

        print(f"    Loaded {class_images:4d} images  ←  {class_name}")

    print(f"\n[INFO] Total images loaded: {len(images)}")
    return np.array(images, dtype="float32"), labels


# ─────────────────────────────────────────────
#  STEP 2 – PREPROCESS
# ─────────────────────────────────────────────
def preprocess(images, labels):
    """
    Normalize pixel values, encode labels as integers,
    then one-hot encode for softmax output.
    """
    # Normalize to [0, 1]
    images = images / 255.0

    # Integer encode labels
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)           # e.g.  "A" → 0, "B" → 1 …
    labels_cat = to_categorical(labels_enc)          # one-hot vectors

    return images, labels_cat, le


# ─────────────────────────────────────────────
#  STEP 3 – BUILD CNN MODEL
# ─────────────────────────────────────────────
def build_model(num_classes):
    """
    A compact but effective CNN suitable for sign-language recognition.
    Architecture: 3× Conv-BN-Pool blocks → Flatten → Dense → Dropout → Softmax
    """
    model = models.Sequential([

        # ── Block 1 ──
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=(*IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 2 ──
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 3 ──
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # ── Classifier ──
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n[MODEL SUMMARY]")
    model.summary()
    return model


# ─────────────────────────────────────────────
#  STEP 4 – TRAIN
# ─────────────────────────────────────────────
def train(model, X_train, y_train, X_val, y_val):
    """Train with data augmentation + smart callbacks."""

    # Real-time augmentation on training data
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_DIR, "sign_language_model.h5")

    callbacks = [
        # Save the best weights automatically
        ModelCheckpoint(best_model_path, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        # Stop early if no improvement after 7 epochs
        EarlyStopping(monitor='val_accuracy', patience=7,
                      restore_best_weights=True, verbose=1),
        # Reduce LR when stuck
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    return history


# ─────────────────────────────────────────────
#  STEP 5 – PLOT TRAINING CURVES
# ─────────────────────────────────────────────
def plot_history(history):
    """Save accuracy & loss curves to model/ folder."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'],     label='Train Acc', color='steelblue')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc',   color='tomato')
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'],     label='Train Loss', color='steelblue')
    axes[1].plot(history.history['val_loss'], label='Val Loss',   color='tomato')
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved → {plot_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("   Sign Language Model Training")
    print("=" * 55)

    # ── 1. Load ──
    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Dataset folder '{DATA_DIR}' not found.")
        print("        Run collect_data.py first to gather training images.")
        return

    images, labels = load_dataset(DATA_DIR)
    if len(images) == 0:
        print("[ERROR] No images found. Collect data first.")
        return

    # ── 2. Preprocess ──
    images, labels_cat, label_encoder = preprocess(images, labels)
    num_classes = len(label_encoder.classes_)
    print(f"[INFO] Classes ({num_classes}): {list(label_encoder.classes_)}")

    # ── 3. Split ──
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels_cat,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels_cat
    )
    print(f"[INFO] Train: {len(X_train)}  |  Val: {len(X_val)}")

    # ── 4. Build model ──
    model = build_model(num_classes)

    # ── 5. Train ──
    print("\n[INFO] Starting training…")
    history = train(model, X_train, y_train, X_val, y_val)

    # ── 6. Evaluate ──
    print("\n[INFO] Evaluating on validation set…")
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"[RESULT] Validation Accuracy: {acc * 100:.2f}%  |  Loss: {loss:.4f}")

    # Detailed report
    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred,
                                  target_names=label_encoder.classes_))

    # ── 7. Save metadata ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    metadata = {
        "classes":       list(label_encoder.classes_),
        "num_classes":   num_classes,
        "img_size":      list(IMG_SIZE),
        "val_accuracy":  float(acc),
    }
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[INFO] Metadata saved → {meta_path}")

    # ── 8. Plot curves ──
    plot_history(history)

    print("\n" + "=" * 55)
    print("[SUCCESS] Training complete!")
    print(f"          Model saved → {MODEL_DIR}/sign_language_model.h5")
    print("=" * 55)


if __name__ == "__main__":
    main()
