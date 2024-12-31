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

