# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 15:42:33 2025

@author: surface
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ========= PARAMETRES FACILES A MODIFIER =========
IMG_SIZE = (128, 128)     # (64,64) plus rapide ; (128,128) souvent mieux
BATCH_SIZE = 32
EPOCHS = 20

FILTERS1 = 32
FILTERS2 = 64
DENSE_UNITS = 128
DROPOUT = 0.3
# ================================================

def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        layers.Conv2D(FILTERS1, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(FILTERS2, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(DENSE_UNITS, activation="relu"),
        layers.Dropout(DROPOUT),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Data augmentation + split validation automatique
train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.1
)

train_data = train_gen.flow_from_directory(
    "data/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_data = train_gen.flow_from_directory(
    "data/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

model = build_cnn()
model.summary()

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save("cnn_cats_dogs.h5")
print("Modèle sauvegardé : cnn_cats_dogs.h5")
print("Classes :", train_data.class_indices)
