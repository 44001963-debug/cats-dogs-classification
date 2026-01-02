# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 10:31:14 2025

@author: surface
"""

# simple_nn_photos.py

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# ----------------------------
# 1. Chargement des images
# ----------------------------

def load_images_from_folder(folder, label, image_size=(32, 32), max_images=None):
    """
    folder : dossier (cats ou dogs)
    label  : 0 pour chat, 1 pour chien
    image_size : taille à laquelle on redimensionne (32, 32)
    max_images : pour limiter le nombre d'images (ex: 200)
    """
    X = []
    y = []
    count = 0

    for filename in os.listdir(folder):
        if max_images is not None and count >= max_images:
            break

        # on ne garde que les images
        if not (filename.lower().endswith(".jpg") or
                filename.lower().endswith(".jpeg") or
                filename.lower().endswith(".png")):
            continue

        path = os.path.join(folder, filename)
        try:
            img = Image.open(path).convert("L")   # "L" = niveaux de gris
            img = img.resize(image_size)          # ex: 32x32
            arr = np.array(img, dtype=np.float32) / 255.0  # normalisation 0-1
            arr_flat = arr.flatten()              # on aplati en vecteur
            X.append(arr_flat)
            y.append(label)
            count += 1
        except Exception as e:
            print("Erreur avec l'image:", path, "->", e)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    return X, y


def load_cat_dog_dataset(base_path="data", image_size=(32, 32), max_per_class=200):
    """
    Charge les dossiers:
       base_path/train/cats
       base_path/train/dogs
    et renvoie X_train, X_val, y_train, y_val
    """
    train_cats_path = os.path.join(base_path, "train", "cats")
    train_dogs_path = os.path.join(base_path, "train", "dogs")

    print("Chargement des CHATS depuis :", train_cats_path)
    X_cats, y_cats = load_images_from_folder(
        train_cats_path, label=0,
        image_size=image_size,
        max_images=max_per_class
    )

    print("Chargement des CHIENS depuis :", train_dogs_path)
    X_dogs, y_dogs = load_images_from_folder(
        train_dogs_path, label=1,
        image_size=image_size,
        max_images=max_per_class
    )

    X = np.vstack([X_cats, X_dogs])
    y = np.vstack([y_cats, y_dogs])

    # on mélange et on fait un split train / validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    return X_train, X_val, y_train, y_val


# ----------------------------
# 2. Réseau de neurones simple
# ----------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)


class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.1):
        # initialisation des poids
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))
        self.lr = learning_rate

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = sigmoid(Z2)
        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        loss = - (1/m) * np.sum(
            y_true * np.log(y_pred_clipped) +
            (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        return loss

    def backward(self, cache, y_true):
        X = cache["X"]
        Z1 = cache["Z1"]
        A1 = cache["A1"]
        A2 = cache["A2"]
        m = X.shape[0]

        dA2 = A2 - y_true
        dW2 = (A1.T @ dA2) / m
        db2 = np.sum(dA2, axis=0, keepdims=True) / m

        dA1 = dA2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=20, print_every=1):
        for epoch in range(epochs):
            y_pred, cache = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(cache, y)
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

    def predict(self, X, threshold=0.5):
        y_pred, _ = self.forward(X)
        return (y_pred >= threshold).astype(int)


# ----------------------------
# 3. Programme principal
# ----------------------------

if __name__ == "__main__":
    # Charger les images depuis data/train/cats et data/train/dogs
    X_train, X_val, y_train, y_val = load_cat_dog_dataset(
        base_path="data",
        image_size=(32, 32),
        max_per_class=200    # tu peux augmenter si ton PC tient
    )

    print("X_train shape :", X_train.shape)
    print("y_train shape :", y_train.shape)
    print("X_val shape   :", X_val.shape)
    print("y_val shape   :", y_val.shape)

    input_dim = X_train.shape[1]   # 32*32 = 1024
    hidden_dim = 64
    output_dim = 1

    model = SimpleNN(input_dim, hidden_dim, output_dim, learning_rate=0.1)

    print("\n--- Entraînement du modèle ---")
    model.train(X_train, y_train, epochs=20, print_every=1)

    print("\n--- Évaluation sur le set de validation ---")
    y_pred_val = model.predict(X_val)
    accuracy = (y_pred_val == y_val).mean()
    print("Accuracy validation :", accuracy)
