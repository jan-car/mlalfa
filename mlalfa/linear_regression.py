# -*- coding: utf-8 -*-
# 2021 Machine Learning Course Alfatraining
# Author: J. Caron
#
# Implementation from Frochte - Maschinelles Lernen - 5.2
#

import logging
import numpy as np


class LinearRegressor(object):

    _log = logging.getLogger(__name__ + '.LinearRegressor')

    def fit(self, X_train, y_train):
        self.n_features = X_train.shape[1]
        self.n_features += 1  # we also want to fit an additional intercept!
        D = self._create_design_matrix(X_train, determine_normalizers=True)
        coeffs_ls, residuals, rank_D, sv_D = np.linalg.lstsq(D, y_train)
        self.coeffs_ls = coeffs_ls  # least-square fitted coefficients!
        self.residuals = residuals  # residuals ???
        self.rank_D = rank_D  # rank of the design matrix A!
        self.sv_D = sv_D  # singular values of the design matrix A!
        self._log.debug(f'\nrank_D: {rank_D}')

    def predict(self, X):
        D = self._create_design_matrix(X)
        y_predict = D @ self.coeffs_ls
        return y_predict

    def _create_design_matrix(self, X, determine_normalizers=False):
        self._log.debug(f'\nX.shape: {X.shape}')
        n_samples = X.shape[0]
        D = np.ones((n_samples, self.n_features))
        self._log.debug(f'\nD.shape: {D.shape}')
        D[:, 1:] = X  # fill matrix with features (leave out 1. column for intercept)!
        if determine_normalizers:
            self.normalizers = D.max(axis=0)  # shape: (n_features,)
        D /= self.normalizers  # divide by maxima along sample axis!
        return D
