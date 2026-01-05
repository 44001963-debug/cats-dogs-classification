# Classification d’images Chats / Chiens
## Optimisation de réseaux de neurones par algorithmes génétiques

## Description du projet
Ce projet a pour objectif de développer un système de classification d’images capable de distinguer automatiquement des images de chats et de chiens.
Nous utilisons des réseaux de neurones convolutifs (CNN) entraînés sur un jeu de données d’images, ainsi que des algorithmes génétiques afin d’optimiser automatiquement les architectures des réseaux et leurs hyperparamètres.

L’objectif principal est d’analyser comment l’évolution automatique des architectures peut améliorer la performance d’un modèle de classification tout en limitant le temps de calcul.

---

## Structure du projet
- `data/` : contient le jeu de données (images de chats et de chiens)
- `ga_cnn.py` : recherche de la meilleure architecture CNN par algorithme génétique
- `train_best_long.py` : entraînement prolongé du meilleur modèle trouvé
- `evaluate.py` : évaluation finale du modèle
- `README.md` : instructions pour exécuter le projet

---

## Méthodes utilisées
- Réseaux de neurones convolutifs (CNN) pour la classification d’images
- Algorithmes génétiques pour faire évoluer automatiquement :
  - la taille des images d’entrée
  - le nombre de couches convolutives
  - le nombre de filtres
  - le taux de dropout
  - le learning rate

Chaque architecture est évaluée sur un ensemble de validation afin de sélectionner la plus performante.

---

## Prérequis
- Python 3.8 ou supérieur
- Bibliothèques Python :
  - tensorflow
  - numpy
  - matplotlib
  - scikit-learn
  - pillow

Installation des dépendances :
```bash
pip install -r requirements.txt
