import numpy as np
import pandas as pd
import missingno as msno
#get the wine dataset from sklearn and take a look at the description provided
from sklearn import datasets

#charger et visualiser dataset
wine = datasets.load_wine()
print(wine.DESCR)

#visualiser dataframe avec la cible et afficher les 5 premieres lignes
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['label'] = wine.target
df.head()