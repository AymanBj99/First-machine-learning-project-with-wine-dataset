import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score



# Charger le dataset wine
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Convertir les données en DataFrame pour faciliter l'analyse
df = pd.DataFrame(X, columns=wine.feature_names)

# Ajouter la cible (type de vin) dans le DataFrame pour faciliter l'exploration
df['target'] = y

# 1. Statistiques descriptives
print("Statistiques descriptives :")
print(df.describe())


# 2. Distribution des variables : histogrammes
df.iloc[:, :-1].hist(bins=20, figsize=(15, 10))
plt.suptitle('Distribution des caractéristiques des vins')
plt.show()

# 3. Matrice de corrélation des caractéristiques
corr_matrix = df.iloc[:, :-1].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matrice de corrélation des caractéristiques')
plt.show()

# 4. Visualisation des classes cibles
sns.countplot(x='target', data=df, palette='Set2')
plt.title('Distribution des classes cibles')
plt.xlabel('Classe')
plt.ylabel('Nombre d\'échantillons')
plt.show()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Normalisation des caractéristiques


# Séparer les données en ensembles d'entraînement (70%) et de test (30%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Créer un modèle KNN avec 5 voisins
knn = KNeighborsClassifier(n_neighbors=5)

# Entraîner le modèle
knn.fit(X_train, y_train)


# Faire des prédictions sur l'ensemble de test
y_pred = knn.predict(X_test)

# Afficher le rapport de classification
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Afficher la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.xlabel('Prédictions')
plt.ylabel('Véritables labels')
plt.title('Matrice de Confusion - KNN')
plt.show()

# Tester différentes valeurs de K (1 à 20)
k_range = range(1, 21)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

# Affichage des résultats
plt.figure(figsize=(8, 6))
plt.plot(k_range, k_scores, marker='o')
plt.xlabel('Nombre de voisins K')
plt.ylabel('Précision moyenne')
plt.title('Recherche du meilleur K pour KNN')
plt.show()

# Meilleur K
best_k = k_range[k_scores.index(max(k_scores))]
print(f"Meilleur K : {best_k} avec une précision moyenne de {max(k_scores):.2f}")