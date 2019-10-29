# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
missingvalues  = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
# strategy: mean / median / most_frequent / constant
# for using constant: (strategy=constant, fill_value =string/numerical)
missingvalues = missingvalues.fit(X[:, 1:3])
X[:, 1:3] = missingvalues.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Encoding the Dependent Variable
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)
y = LabelEncoder().fit_transform(y)
