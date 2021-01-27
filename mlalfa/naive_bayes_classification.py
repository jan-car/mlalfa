# -*- coding: utf-8 -*-
# 2021 Machine Learning Course Alfatraining
# Author: J. Caron
#
# Implementation from Frochte - Maschinelles Lernen - 4.3.2
#

import logging
import numpy as np
from numpy.core.fromnumeric import shape


class NaiveBayesNominalEstimator(object):

    _log = logging.getLogger(__name__ + '.NaiveBayesNominalEstimator')

    def __init__(self, vectorized=True):
        if vectorized:
            self.predict = self._predict_vectorized
        else:
            self.predict = self._predict_nonvectorized

    def fit(self, XTrain, yTrain):
        self._log.debug(f'\nXTrain.shape: {XTrain.shape}')
        self._log.debug(f'\nYTrain.shape: {YTrain.shape}')
        # PXI: Probability of X(symptom/feature) under the assumption of I (diagnosis/label)
        PXI = np.zeros((2, XTrain.shape[1], 2))
        self._log.debug(f'\nPXI.shape: {PXI.shape}')
        # Shape: (diagnosis is True/False, the symptoms, symptom is True/False)
        for k in range(XTrain.shape[1]):
            # Number of patients with symptom k, who are diagnosed as ill:
            PXI[1, k, 1] = np.sum(np.logical_and(XTrain[:, k], yTrain))
            # Number of patients that don't show symptom k, who are diagnosed as ill:
            PXI[1, k, 0] = np.sum(np.logical_and(np.logical_not(XTrain[:, k]), yTrain))
            # Number of patients that show symptom k, who are not diagnosed as ill:
            PXI[0, k, 1] = np.sum(np.logical_and(XTrain[:, k], np.logical_not(yTrain)))
            # Number of patients that don't show symptom k, who are not diagnosed as ill:
            PXI[0, k, 0] = np.sum(np.logical_not(np.logical_or(XTrain[:, k], yTrain)))
            # DeMorgan: not A & not B = not(A or B)
        # Turn into probability (+1/2 makes sure we don't have zeros for divisions later!)
        counts_record = XTrain.shape[0]  # Number of lines / records / entries in the training data!
        PXI = (PXI + 1/2) / (counts_record + 1)
        # Probability of diagnosis overall:
        PI = np.zeros(2)
        PI[1] = np.sum(yTrain)
        PI[0] = counts_record - PI[1]
        PI = PI / counts_record  # now it's a probability!
        self.PXI = PXI
        self.PI = PI

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
        X = X.astype(int)
        X = np.logical_not(np.stack((X, np.logical_not(X)), axis=-1))
        X = X[:, None, ...]
        self._log.debug(f'\nX.shape: {X.shape}')
        P = np.zeros_like(self.PI)  # probability that new patient has illness or not


        self._log.debug(f'\n{self.PXI.shape}, *, {X.shape}')
        temp = self.PXI * X
        self._log.debug(f'\ntemp: {temp.shape}\n')
        temp2 = temp.sum(axis=-1)
        self._log.debug(f'\ntemp2: {temp2.shape}\n')
        P = np.prod(temp2, axis=-1)

        # self._log.debug((np.prod(self.PXI * X, axis=(-1, -2))).shape)
        # P = np.prod(self.PXI * X, axis=(-1, -2)) * self.PI  # Bayes Rule counter!
        #self._log.debug(f'\nP: {P.shape}\n{P}')
        denominator = np.sum(P)  # Bayes Rule denominator!
        P = P/denominator  # Combined! Denotes probability of patient [being ill, being not ill]
        y = np.argmax(P, axis=1)  # Choose the option that's more likely!
        return y
