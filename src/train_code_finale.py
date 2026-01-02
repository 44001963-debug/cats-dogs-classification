# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 15:16:00 2025

@author: surface
"""
# train_transfer.py
# Transfer Learning Cats vs Dogs (MobileNetV2)
# - split train/val automatique
# - entraînement rapide sur CPU
# - 2 phases : (1) tête seule, (2) fine-tuning partiel
# - sauvegarde meilleur modèle en .keras

import os
import time
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ================== PARAMETRES A MODIFIER ==================
DATA_DIR = "data/train"     # doit contenir data/train/cats et data/train/dogs
VAL_SPLIT = 0.2
SEED = 42

IMG_SIZE = (160, 160)       # 128 plus rapide, 160 bon compromis, 224 souvent mieux mais + lent
BATCH_SIZE = 32

EPOCHS_HEAD = 8             # Phase 1 (tête)
EPOCHS_FT = 10              # Phase 2 (fine-tuning)

LR_HEAD = 1e-3
LR_FT = 1e-4

DROPOUT = 0.25

# Pour corriger le biais "tout chat" : pénaliser plus les erreurs sur chiens (label 1)
USE_CLASS_WEIGHT = True
CLASS_WEIGHT = {0: 1.0, 1: 1.25}   # augmente 1.25 -> 1.4 si dogs->cats trop fréquent

MODEL_OUT = "tl_mobilenetv2_best.keras"
# ============================================================


def build_model(img_size):
    # Base pré-entraînée ImageNet
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False  # Phase 1: on gèle

    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    # MobileNetV2 attend des pixels prétraités:
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


def make_gens(img_size, batch_size, val_split, seed):
    # Data augmentation légère (utile + pas trop lente)
    gen = ImageDataGenerator(
        validation_split=val_split,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.05,
        height_shift_range=0.05
    )

    train_data = gen.flow_from_directory(
        DATA_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=seed
    )

    val_data = gen.flow_from_directory(
        DATA_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )

    return train_data, val_data


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    start_total = time.perf_counter()

    print("\n=== Chargement dataset ===")
    train_data, val_data = make_gens(IMG_SIZE, BATCH_SIZE, VAL_SPLIT, SEED)
    print("Classes :", train_data.class_indices)

    print("\n=== Build modèle (MobileNetV2) ===")
    model, base = build_model(IMG_SIZE)
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
    ]

    # ---------------- Phase 1 : tête (base gelée) ----------------
    print("\n=== Phase 1 : entraînement tête (base gelée) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_HEAD),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    cw = CLASS_WEIGHT if USE_CLASS_WEIGHT else None

    t1 = time.perf_counter()
    hist1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
        class_weight=cw,
        verbose=1
    )
    t2 = time.perf_counter()
    print(f"Temps phase 1: {t2 - t1:.2f}s")

    # ---------------- Phase 2 : fine-tuning partiel ----------------
    print("\n=== Phase 2 : fine-tuning partiel ===")
    # On dégèle seulement la fin du réseau (plus stable / pas trop lent)
    base.trainable = True

    # Débloquer seulement les dernières couches (ex: dernières ~30 couches)
    fine_tune_at = max(0, len(base.layers) - 30)
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FT),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    t3 = time.perf_counter()
    hist2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_FT,
        callbacks=callbacks,
        class_weight=cw,
        verbose=1
    )
    t4 = time.perf_counter()
    print(f"Temps phase 2: {t4 - t3:.2f}s")

    end_total = time.perf_counter()
    print("\n=== FIN ===")
    print(f"Meilleur modèle sauvegardé : {MODEL_OUT}")
    print(f"Temps total: {end_total - start_total:.2f}s")


if __name__ == "__main__":
    main()

