# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 22:11:26 2025

@author: surface
"""
# -*- coding: utf-8 -*-
"""
evaluate_best_long_v2.py
Évalue un modèle cats vs dogs avec :
- accuracy + confusion matrix + classification report
- recherche du MEILLEUR SEUIL (au lieu de 0.5 fixe)
- comptage exact correct/incorrect
- temps d'inférence
- exemples d'erreurs

IMPORTANT :
- IMG_SIZE doit être la même taille que celle utilisée à l'entraînement du modèle.
- MODEL_PATH doit pointer vers ton modèle (.keras ou .h5)
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# =========================
# PARAMÈTRES À AJUSTER
# =========================
DATA_DIR = "data/train"          # doit contenir cats/ et dogs/
VAL_SPLIT = 0.2
BATCH_SIZE = 32

MODEL_PATH = "best_long.keras"   # modèle entraîné
IMG_SIZE = (128, 128)            # même taille que l'entraînement
TOP_ERRORS = 10                  # nb d'erreurs affichées

# Seuil :
# - Si AUTO_THRESHOLD=True, on cherche le meilleur seuil sur validation (0.05..0.95)
# - Sinon on utilise FIXED_THRESHOLD
AUTO_THRESHOLD = True
FIXED_THRESHOLD = 0.5

SEED = 42
# =========================

np.random.seed(SEED)
tf.random.set_seed(SEED)

def pick_best_threshold(y_true: np.ndarray, proba: np.ndarray):
    best_t = 0.5
    best_acc = -1.0
    best_cm = None
    for t in np.linspace(0.05, 0.95, 19):
        y_pred = (proba >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
            best_cm = confusion_matrix(y_true, y_pred)
    return best_t, best_acc, best_cm

# 1) Dataset validation
gen = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)

val_data = gen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# 2) Charger le modèle
model = tf.keras.models.load_model(MODEL_PATH)

# 3) Prédictions (proba)
t0 = time.perf_counter()
proba = model.predict(val_data, verbose=0).ravel()
t1 = time.perf_counter()

y_true = val_data.classes

# Debug utile : voir si le modèle sort toujours ~0.4 etc.
print("\n===== DEBUG PROBAS =====")
print(f"proba min={proba.min():.4f}  max={proba.max():.4f}  mean={proba.mean():.4f}  std={proba.std():.4f}")

# 4) Choix du seuil
if AUTO_THRESHOLD:
    best_t, best_acc, best_cm = pick_best_threshold(y_true, proba)
    threshold = best_t
    print("\n===== SEUIL AUTOMATIQUE =====")
    print(f"Meilleur seuil trouvé: {threshold:.2f}  => accuracy={best_acc:.3f}")
else:
    threshold = FIXED_THRESHOLD
    print("\n===== SEUIL FIXE =====")
    print(f"Seuil utilisé: {threshold:.2f}")

# 5) Résultats finaux avec ce seuil
y_pred = (proba >= threshold).astype(int)

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

report = classification_report(
    y_true, y_pred,
    target_names=["cats", "dogs"],
    digits=3,
    zero_division=0
)

print(f"\nTemps prédiction (inférence): {t1 - t0:.2f} s")
print("\n===== RÉSULTATS =====")
print(f"Accuracy: {acc:.3f}")

print("\nConfusion matrix (format sklearn):")
print(cm)

print("\nInterprétation (si cats=0, dogs=1):")
print(f"Chats bien classés (cats->cats): {cm[0,0]}")
print(f"Chats mal classés  (cats->dogs): {cm[0,1]}")
print(f"Chiens mal classés (dogs->cats): {cm[1,0]}")
print(f"Chiens bien classés(dogs->dogs): {cm[1,1]}")

correct = int((y_pred == y_true).sum())
total = int(len(y_true))
wrong = total - correct
print(f"\nTotal correct: {correct}/{total}")
print(f"Total wrong  : {wrong}/{total}")

print("\n===== CLASSIFICATION REPORT =====")
print(report)

print("Classes (dossier -> label):", val_data.class_indices)

# 6) Exemples d'erreurs
wrong_idx = np.where(y_pred != y_true)[0]
if len(wrong_idx) == 0:
    print("\nAucune erreur ✅")
else:
    print(f"\n===== EXEMPLES D'ERREURS (top {min(TOP_ERRORS, len(wrong_idx))}) =====")
    filepaths = np.array(val_data.filepaths)
    for i in wrong_idx[:TOP_ERRORS]:
        true_label = "dogs" if y_true[i] == 1 else "cats"
        pred_label = "dogs" if y_pred[i] == 1 else "cats"
        print(f"- {filepaths[i]} | true={true_label} pred={pred_label} proba_dog={proba[i]:.3f}")
