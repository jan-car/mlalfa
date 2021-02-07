# -*- coding: utf-8 -*-

# %% [markdown]
# Stufe 1
# Basteles Dir ein minimales Beipiel für die Anwendung des SimpleImputers. (Ähnlich wie Doku)
# Vollziehe die Aktionen mit den Beispielen nach
#
# Schau die die Seiten der Mathebibel an und vollziehe das Beispiel mit dem Eigenwert nach.
# https://www.mathebibel.de/eigenwerte-eigenvektoren
#
# Stufe 2
# Generiere Dir Daten für eine drei-dimensionale Struktur der Form a*x1+b*x2+c*x3=y.
# Falls erforderlich, zentriere die Daten
# Füge für die zweite Achse viel Rauschen hinzu und plotte.
# Transformiere die X-Daten, so dass sie durch das neue Koordinatensystem beschrieben werden.
# Ermittle die neuen Koordinatenvektoren und die erklärte Varianz  mittels PCA.
# Transformiere die X-Daten, so dass sie durch das neue Koordinatensystem beschrieben werden
# und schaue die transformierten Daten an.
#
# Stufe 3a
# Führe eine Lineare Regression auf den Daten vor der PCA durch,
# führe eine PCA durch und reduziere die Dimensionen und mache dann erneut eine PCA.
# Vergleiche den Plot.
#
# alternativ 3b :
# Beschäftige dich mit SVM von gestern.
#
# Stufe 4
# importiere sklearn.datasets.make_swissrole und überlege, ob man hier eine KernelPCA ansetzen könnte.
#


# %% [markdown]
# # Imputer stuff:
# %%
import numpy as np
from numpy.core.defchararray import replace
from sklearn.impute import SimpleImputer

n = 9
incomplete_array = np.arange(n**2, dtype=float).reshape((n, n))
ii = np.random.choice(np.arange(n), size=n, replace=False)
jj = np.random.choice(np.arange(n), size=n, replace=False)
for i, j in zip(ii, jj):
    incomplete_array[i, j] = np.nan
print(incomplete_array)

imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
print(imputer_mean.fit_transform(incomplete_array))


# %% [markdown]
# # Python Data Science Handbook (VanDerPlas) PCA in Depth
# See https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

# %% [1]
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# %% [markdown]
# ## Introducing Principal Component Analysis:

# %% [2]
np.random.seed(6)
x1 = np.random.rand(2, 2)
x2 = np.random.rand(2, 200)
X = np.dot(x1, x2).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal')
# %% [3]
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
# %% [4]
print(pca.components_)
# %% [5]
print(pca.explained_variance_)
# %% [6] modified
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# plot data
axes[0].scatter(X[:, 0], X[:, 1], alpha=0.5)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * np.sqrt(length)
    axes[0].arrow(*pca.mean_, *v, width=0.01, color='black')
axes[0].axis('equal')
axes[0].set(xlabel='x', ylabel='y', title='input')

# %% [markdown]
# ## PCA as dimensionality reduction:
# %% [from Appendix]
# plot principal components
X_pca = pca.transform(X)
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
axes[1].arrow(*(0, 0), *(0, 0.5), width=0.01, color='black')
axes[1].arrow(*(0, 0), *(0.5, 0), width=0.01, color='black')
axes[1].axis('equal')
axes[1].set(xlabel='component 1', ylabel='component 2', title='principal components')#, xlim=(-1, 1), ylim=(-0.2, 0.2))
# %% [7]
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)
# %% [8]
X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.5)
plt.axis('equal')


# %% [Additions for Elisabeths tasks]
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

X_train, y_train = X[:, :1], X[:, 1]  # shapes: (n_samples, 1), (n_samples,)
X_train_new, y_train_new = X_new[:, :1], X_new[:, 1]  # shapes: (n_samples, 1), (n_samples,)
x_plot = np.linspace(X_train.min(), X_train.max(), 100)

model = LinearRegression().fit(X_train, y_train)
y_predict = model.predict(x_plot[:, None])

model_pca = make_pipeline(PCA(n_components=1), LinearRegression()).fit(X_train_new, y_train_new)
y_predict_pca = model_pca.predict(x_plot[:, None])

plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.75)
plt.plot(x_plot, y_predict, linewidth=2, alpha=0.75, color='g', label='linear fit before pca')
plt.plot(x_plot, y_predict_pca, linewidth=2, linestyle='--', alpha=0.5, color='r', label='linear fit after pca')
plt.axis('equal')
plt.legend()
