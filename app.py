import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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

