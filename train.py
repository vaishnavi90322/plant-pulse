"""
train.py  ─  Plant Pulse | MobileNetV2 Transfer-Learning Trainer
================================================================
Usage:
    python train.py --data_dir path/to/dataset

Dataset folder must look like:
    dataset/
        Apple___Apple_scab/
            img1.jpg
            img2.jpg
            ...
        Apple___Black_rot/
            ...
        Tomato___healthy/
            ...
        (38 class folders total)

After training, the model is saved as:
    mobilenetv2_best.keras   ← placed in the same folder as app.py
"""

import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import datetime

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train MobileNetV2 on plant disease dataset")
parser.add_argument("--data_dir",  required=True, help="Path to dataset root folder")
parser.add_argument("--epochs",    type=int, default=20,  help="Max training epochs (default 20)")
parser.add_argument("--batch",     type=int, default=32,  help="Batch size (default 32)")
parser.add_argument("--img_size",  type=int, default=224, help="Image size (default 224)")
parser.add_argument("--val_split", type=float, default=0.15, help="Validation split (default 0.15 = 15%%)")
parser.add_argument("--fine_tune", action="store_true",   help="Add a fine-tuning phase after initial training")
args = parser.parse_args()

IMG_SIZE   = (args.img_size, args.img_size)
BATCH      = args.batch
EPOCHS     = args.epochs
DATA_DIR   = args.data_dir
SAVE_PATH  = os.path.join(os.path.dirname(__file__), "mobilenetv2_best.keras")
LOG_DIR    = os.path.join(os.path.dirname(__file__), "logs",
                          datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

print(f"\n{'='*60}")
print(f"  Plant Pulse — MobileNetV2 Trainer")
print(f"{'='*60}")
print(f"  Dataset  : {DATA_DIR}")
print(f"  Img size : {IMG_SIZE}")
print(f"  Batch    : {BATCH}")
print(f"  Epochs   : {EPOCHS}")
print(f"  Val split: {args.val_split*100:.0f}%")
print(f"  Save to  : {SAVE_PATH}")
print(f"{'='*60}\n")

# ── 1. Data loading ───────────────────────────────────────────────────────────
print("Loading dataset …")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=args.val_split,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=args.val_split,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=False,
)

CLASS_NAMES = train_ds.class_names
NUM_CLASSES = len(CLASS_NAMES)
print(f"\nDetected {NUM_CLASSES} classes:")
for i, c in enumerate(CLASS_NAMES):
    print(f"  [{i:02d}] {c}")
print()

# ── 2. Performance tuning for data pipeline ───────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# ── 3. Augmentation (applied only during training) ────────────────────────────
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1),
], name="augmentation")

# ── 4. Build model ────────────────────────────────────────────────────────────
print("Building model …")
base = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMG_SIZE, 3),
)
base.trainable = False   # freeze base for initial training

inputs  = layers.Input(shape=(*IMG_SIZE, 3))
x       = data_aug(inputs)
x       = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x       = base(x, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dropout(0.3)(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs, name="PlantPulse_MobileNetV2")
model.summary()

# ── 5. Compile ────────────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ── 6. Callbacks ──────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
    TensorBoard(log_dir=LOG_DIR),
]

# ── 7. Phase 1 — Train only top layers ────────────────────────────────────────
print("\n── Phase 1: Training top layers (base frozen) ──\n")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ── 8. Phase 2 — Fine-tune (optional, --fine_tune flag) ──────────────────────
if args.fine_tune:
    print("\n── Phase 2: Fine-tuning top 50 layers of MobileNetV2 ──\n")
    base.trainable = True
    # Freeze all layers except the last 50 in the base
    for layer in base.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
    )

# ── 9. Final report ───────────────────────────────────────────────────────────
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"\n{'='*60}")
print(f"  ✅ Training complete!")
print(f"  Val Accuracy : {val_acc*100:.2f}%")
print(f"  Val Loss     : {val_loss:.4f}")
print(f"  Model saved  : {SAVE_PATH}")
print(f"{'='*60}\n")
print("Now restart app.py — the real AI model will be used automatically.\n")
