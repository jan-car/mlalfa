# -*- coding: utf-8 -*-

# %% [markdown]
# Aufgabe1a:
# Hole dir die Daten "Breast cancer wisconsin dataset" und schaue dir die Daten an
# Tipp: sklearn.datasets enthält die Beschreibung und eine Funktion zum Laden der Daten
#
# Aufgabe 1b:
# Skaliere die Daten
# Variiere 2 Parameter und beurteile den Effekt
#
# Aufgabe 2
# Hole dir die Daten des Boston housing set
# Tipp: sklearn.datasets enthält die Beschreibung und eine Funktion zum Laden der Daten
#
# Skaliere die Daten mit Hilfe der Klasse sklearn.preprocessing.MinMaxScaler
# Versuche mit einer vordefinierten MLP-Klasse den Preis vorauszusagen.
# Variere die Parameter der MLP-Klasse, um ein besseres Ergebnis zu bekommen.
#
# Aufgabe 3a
# Skaliere die Daten mit Hilfe der Klasse sklearn.preprocessing.StandardScaler
# Definiere ein Raster für die gewünschten Parameterkombinationen.
# Verwirkliche alle Kombinationen in einer Schleife, in der jeweils einzelne Estimators aufgerufen werden.
#
# Aufgabe 3b
# Versuche im Frochte Code die mathematischen Formeln wiederzufinden.
# Versuche


# %% [markdown]
# # Imports:

# %%
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import pandas as pd
from IPython import get_ipython
if type(get_ipython()).__name__ == 'ZMQInteractiveShell':  # IPython Notebook!
    get_ipython().run_line_magic('matplotlib', 'inline')  # '%matplotlib inline'
    get_ipython().run_line_magic('load_ext', 'autoreload')  # '%load_ext autoreload'
    get_ipython().run_line_magic('autoreload', '2')  # '%autoreload 2' (reloads everything)
# Change working directory to file location:
os.chdir(R'C:\Users\Jan\Projects\mlalfa\scripts')
_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
np.random.seed(42)


# %% [markdown]
# # Breast Cancer Dataset:
# %%
# Load data and print info:
data = load_breast_cancer()
print(data.keys())
print(data.DESCR)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# Show as dataframe:
df = pd.DataFrame(X, columns=data['feature_names'])
df.head()
# %%
# And describe:
df.describe()
# %%
# Scale data and view as dataframe:
scaler = StandardScaler(with_mean=True, with_std=True)  # Both are defaults!
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=data['feature_names'])
df_scaled.head()
# %%
# And describe:
df_scaled.describe()

# %% [markdown]
# # Multi-Layer-Perceptron Classifier:
# %%
network = MLPClassifier(hidden_layer_sizes=(50, 50),  # default: (100, )
                        activation='relu',  # default: 'relu'
                        solver='adam',  # default: 'adam'
                        batch_size='auto',  # default: 'auto'
                        max_iter=200,  # default: 200
                        random_state=42,  # default: 42
                        alpha=0.0001,  # default: 0.0001
                        tol=0.0001,  # default: tol=0.0001
                        early_stopping=True,  # default: False
                        n_iter_no_change=20,  # default: 10
                        verbose=True)  # default: False
model = make_pipeline(scaler, network)
model.fit(X_train, y_train)
# %%
# Show Score:
print(f'\nTrain score: {model.score(X_train, y_train)*100:.3g}%')
print(f'Test  score: {model.score(X_test, y_test)*100:.3g}%')


# %% [markdown]
# # Boston Housing Dataset:
# %%
# Load data and print info:
data = load_boston()
print(data.keys())
print(data.DESCR)
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# Show as dataframe:
df = pd.DataFrame(X, columns=data['feature_names'])
df.head()
# %%
# And describe:
df.describe()
# %%
# Scale data and view as dataframe:
scaler = MinMaxScaler(feature_range=(0, 1))  # defaults!
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=data['feature_names'])
df_scaled.head()
# %%
# And describe:
df_scaled.describe()


# %% [markdown]
# # Multi-Layer-Perceptron Regressor:
# %%
network = MLPRegressor(hidden_layer_sizes=(200, 200, 200),  # default: (100, )
                       activation='relu',  # default: 'relu'
                       solver='adam',  # default: 'adam'
                       batch_size='auto',  # default: 'auto'
                       max_iter=1000,  # default: 200
                       random_state=42,  # default: 42
                       alpha=0.0001,  # default: 0.0001
                       tol=0.0001,  # default: tol=0.0001
                       early_stopping=True,  # default: False
                       n_iter_no_change=20,  # default: 10
                       verbose=True)  # default: False
model = make_pipeline(MinMaxScaler(), network)
model.fit(X_train, y_train)
# %%
# Show Root-Mean-Square-Error:
y_predict_train = model.predict(X_train)
y_predict_test = model.predict(X_test)
print(f'\nTrain RMSE: {np.sqrt(mean_squared_error(y_predict_train, y_train))*1000:.2f}$')
print(f'Test  RMSE: {np.sqrt(mean_squared_error(y_predict_test, y_test))*1000:.2f}$')
print('For comparison (in test data):')
print(f'Highest price: {y_test.max()*1000:.2f}$')
print(f'Lowest  price: {y_test.min()*1000:.2f}$')
# %%
# Plot sorted house pricing and predictions (training data):
train_idx = np.argsort(y_train)
plt.figure(figsize=(8, 5))
plt.plot(y_predict_train[train_idx]*1000, color='black', linewidth=2)
plt.plot(y_train[train_idx]*1000, color='red', linewidth=4, alpha=0.75)
plt.xlabel('House index (sorted acc. to asc. prices)')
plt.ylabel('House Price [$]')
plt.xlim(0, len(y_train))
plt.ylim(0, 55000)
plt.title('Training Data')
# %%
# Plot sorted house pricing and predictions (test data):
test_idx = np.argsort(y_test)
plt.figure(figsize=(8, 5))
plt.plot(y_predict_test[test_idx]*1000, color='black', linewidth=2)
plt.plot(y_test[test_idx]*1000, color='red', linewidth=4, alpha=0.75)
plt.xlabel('House index (sorted acc. to asc. prices)')
plt.ylabel('House Price [$]')
plt.xlim(0, len(y_test))
plt.ylim(0, 55000)
plt.title('Test Data')
