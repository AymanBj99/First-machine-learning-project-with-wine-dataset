import numpy as np
import pandas as pd
import missingno as msno
#get the wine dataset from sklearn and take a look at the description provided
from sklearn import datasets

#charger et visualiser dataset
wine = datasets.load_wine()
print(wine.DESCR)

