import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


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
