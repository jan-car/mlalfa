# -*- coding: utf-8 -*-
# %% [markdown]
# # Imports:

# %%
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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
# # Teil 1: Regressionsbeispiel von vanderPlas (Python Data Science Handbook)
# ## Create training data and plot them:
# %%
x = 10 * np.random.rand(50)
y = 2 * x - 5 + np.random.randn(50)
plt.scatter(x, y)
# %% [markdown]
# ##
# %%

model = LinearRegression(fit_intercept=True)

model.fit(x[:, np.newaxis], y)

xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
# %%
print('Model slope:     ', model.coef_[0])
print('Model intercept: ', model.intercept_)

# %%
poly = PolynomialFeatures(7, include_bias=True)
x = 10 * np.random.rand(50)
y = np.sin(x) + 0.1 * np.random.randn(50)

poly.fit_transform(x[:, None])

model = LinearRegression()

model.fit(poly.transform(x[:, None]), y)

xfit = np.linspace(0, 10, 1000)
yfit = model.predict(poly.transform(xfit[:, None]))

plt.scatter(x, y)
plt.plot(xfit, yfit)
# %%

# %%


