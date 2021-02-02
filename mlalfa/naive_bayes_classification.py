# -*- coding: utf-8 -*-
# 2021 Machine Learning Course Alfatraining
# Author: J. Caron
#
# Implementation from Frochte - Maschinelles Lernen - 4.3.2
# See https://scikit-learn.org/stable/modules/naive_bayes.html for more info (especially Bernoulli Naive Bayes in 1.9.4)
#

import logging
import numpy as np


class NaiveBayesNominalEstimator(object):

    _log = logging.getLogger(__name__ + '.NaiveBayesNominalEstimator')

    def __init__(self, vectorized=True):
        if vectorized:
            self.predict = self._predict_vectorized
        else:
            self.predict = self._predict_nonvectorized

    def fit(self, X_train, y_train):
        self._log.debug(f'\nXTrain.shape: {X_train.shape}')
        self._log.debug(f'\nYTrain.shape: {y_train.shape}')
        # PXI: Probability of X(symptom/feature) under the assumption of I (diagnosis/label)
        PXI = np.zeros((2, X_train.shape[1], 2))
        self._log.debug(f'\nPXI.shape: {PXI.shape}')
        # Shape: (diagnosis is True/False, the symptoms, symptom is True/False)
        for k in range(X_train.shape[1]):
            # Number of patients with symptom k, who are diagnosed as ill:
            PXI[1, k, 1] = np.sum(np.logical_and(X_train[:, k], y_train))
            # Number of patients that don't show symptom k, who are diagnosed as ill:
            PXI[1, k, 0] = np.sum(np.logical_and(np.logical_not(X_train[:, k]), y_train))
            # Number of patients that show symptom k, who are not diagnosed as ill:
            PXI[0, k, 1] = np.sum(np.logical_and(X_train[:, k], np.logical_not(y_train)))
            # Number of patients that don't show symptom k, who are not diagnosed as ill:
            PXI[0, k, 0] = np.sum(np.logical_not(np.logical_or(X_train[:, k], y_train)))
            # DeMorgan: not A & not B = not(A or B)
        # Turn into probability (+1/2 makes sure we don't have zeros for divisions later!)
        counts_record = X_train.shape[0]  # Number of lines / records / entries in the training data!
        PXI = (PXI + 1/2) / (counts_record + 1)
        # Probability of diagnosis overall:
        PI = np.zeros(2)
        PI[1] = np.sum(y_train)
        PI[0] = counts_record - PI[1]
        PI = PI / counts_record  # now it's a probability!
        self.PXI = PXI
        self.PI = PI
        return self

    def _predict_nonvectorized(self, X):
        choosenClasses = []
        for x in X:
            x = x.astype(int)
            P = np.zeros_like(self.PI)  # probability that new patient has illness or not
            allofthem = np.arange(X.shape[1])  # [0, 1, 2, 3, 4]: indices of the symptoms
            for i in range(len(self.PI)):
                P[i] = np.prod(self.PXI[i, allofthem, x]) * self.PI[i]  # Bayes Rule counter!
            denominator = np.sum(P)  # Bayes Rule denominator!
            P = P / denominator  # Combined! Denotes probability of patient [being ill, being not ill]
            choosenClass = np.argmax(P)  # Choose the option that's more likely!
            choosenClasses.append(choosenClass)
        return np.asarray(choosenClasses)

    def _predict_vectorized(self, X):
        self._log.debug(f'\nX: {X.shape}\n')
        X = X.astype(int)  # shape: (r, f) with r: # of records/samples, f: # of symptoms/features
        # Stack X and its logical inverse (not X)
        X = np.stack(((1-X), X), axis=-1)  # shape: (r, f, l), l: (No symptom / Yes symptom)
        self._log.debug(f'\nX.shape: {X.shape}')
        X = X[:, None, :, :]  # shape: (r, 1, f, l)
        self._log.debug(f'\nX.shape: {X.shape}')
        P = np.zeros_like(self.PI)  # probability that new patient has illness or not, shape: (c,), c=2 (# of classes)
        self._log.debug(f'\nP.shape: {P.shape}')
        self._log.debug(f'\n{self.PXI.shape} * {X.shape}')
        self._log.debug(f'\nself.PXI.shape: {self.PXI.shape}')  # PXI shape: (c, f, l)
        weighted_X = self.PXI * X  # shape: (c, f, l) * (r, 1, f, l) = (r, c, f, l)
        self._log.debug(f'\nweighted_X: {weighted_X.shape}')
        weighted_X_sum = weighted_X.sum(axis=-1)  # shape: (r, c, f), sum over l, corresponds to: PIX*X + (1-PIX)*(1-X)
        self._log.debug(f'\nweighted_X_sum: {weighted_X_sum.shape}')
        self._log.debug(f'\nself.PI.shape: {self.PI.shape}')
        P = np.prod(weighted_X_sum, axis=-1) * self.PI  # shape: (r, c), the product was over the features
        self._log.debug(f'\nP.shape: {P.shape}')
        denominator = np.sum(P)  # Bayes Rule denominator! shape: (r, c), different denominators for the c categories!
        self._log.debug(f'\ndenominator.shape: {denominator.shape}')
        P = P/denominator  # Combined! Denotes probability of patient [being ill, being not ill], shape: (r, c)
        y = np.argmax(P, axis=1)  # Choose the option that's more likely! shape: (r,)
        return y
