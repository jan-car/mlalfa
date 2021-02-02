# -*- coding: utf-8 -*-
# 2021 Machine Learning Course Alfatraining
# Author: J. Caron
#
# Implementation from Frochte - Maschinelles Lernen - 7.1
#

import logging
import numpy as np


def heaviside(X):
    """Heaviside activation function"""
    y = np.ones_like(X, dtype=np.float)
    y[X <= 0] = 0
    return(y)


class OneLayerPerceptron(object):

    _log = logging.getLogger(__name__ + '.OneLayerPerceptron')

    def __init__(self, tmax=1000, eta=0.25, activation=heaviside, random_state=42):
        # Save construction parameters:
        self.tmax = tmax  # Maximum iterations
        self.eta = eta  # Learning rate
        self.activation = activation  # Activation function (default: heaviside)
        self.random_state = random_state

    def fit(self, X_train, y_train):
        np.random.seed(self.random_state)
        # Create the design matrix (normalize the features and add a column of ones for the bias at the right):
        X_d = self._create_design_matrix(X_train, determine_normalizers=True)
        n_features = X_d.shape[1]
        # Initialisation:
        Dw = np.zeros(n_features)  # Weight updates initialised as zeros:
        w = np.random.rand(n_features) - 0.5  # Weights randomly in the range [-0.5, 0.5]
        t, convergence = 0, np.inf  # Iteration parameter t and Convergence/costfunction
        while (convergence > 0) and (t < self.tmax):  # Do until total convergence or max iteration number reached:
            t += 1  # Uptick iterator t
            rng_idx = np.random.randint(len(y_train))  # Choose a single sample randomly to train with!
            x_update = X_d[rng_idx, :].T
            y_update = y_train[rng_idx]
            error = y_update - heaviside(w@x_update)  # Calculate the error for the chosen sample!
            if t == 1:  # Only log the first iteration to avoid spam:
                self._log.debug(f'\nchosen_sample: {rng_idx}')
                self._log.debug(f'\nw.shape: {w.shape}')
                self._log.debug(f'\nx_update.shape: {x_update.shape}')
                self._log.debug(f'\ny_update.shape: {y_update.shape}')
                self._log.debug(f'\nerror.shape: {error.shape}')
            for j in range(n_features):  # Iterate over all features!
                Dw[j] = self.eta * error * x_update[j]  # Weight update!
                w[j] = w[j] + Dw[j]  # Updated weight!
            convergence = np.linalg.norm(y_train - self.activation(w@X_d.T))
        self.w = w  # Save the weights!
        self._log.debug(f'\nw: \n{w}')
        self._log.info(f'\nNeeded t={t} steps until convergence ({convergence})')
        return self

    def predict(self, X):
        # Create the design matrix (use normalization from the training data here):
        X_d = self._create_design_matrix(X)
        h = self.w@X_d.T  # Weighted sum!
        return self.activation(h)  # Return predictions!

    def _create_design_matrix(self, X, determine_normalizers=False):
        self._log.debug(f'\nX.shape: {X.shape}')
        # If used in the fit() function, determin_normalizers should be True, False in predict!
        if determine_normalizers:  # Normalize first (otherwise last column with only ones -> nan!)
            self.X_min = X.min(axis=0)
            self.X_max = X.max(axis=0)
        X = (X - self.X_min) / (self.X_max - self.X_min)  # Normalize to min/max of training data!
        X_design = np.ones((X.shape[0], X.shape[1]+1))  # 1 more column for bias term!
        X_design[:, :-1] = X[...]  # Last column is for bias terms, full of ones!
        self._log.debug(f'\nX_design.shape: {X_design.shape}')
        self._log.debug(f'\nX_design: {X_design}')
        return X_design
