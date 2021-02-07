# -*- coding: utf-8 -*-
# 2021 Machine Learning Course Alfatraining
# Author: J. Caron
#
# Implementation from Frochte - Maschinelles Lernen - 5.4
#

import logging
import numpy as np
from scipy.spatial import KDTree


class KNearestNeighborClassifier(object):

    _log = logging.getLogger(__name__ + '.KNearestNeighborClassifier')

    def __init__(self, k=4, smear=1, norm_ord=None, vectorized=True):
        self.k = k  # Number of nearest neighbors!
        self.smear = smear  # Smear of the distances to avoid inverse distances diverging!
        self.norm_ord = norm_ord  # Order of the norm to use!
        if vectorized:
            self.predict = self._predict_vectorized
        else:
            self.predict = self._predict_non_vectorized

    def fit(self, X_train, y_train):
        # Normalize data (over all samples/rows) and save stuff:
        self.x_max = np.max(X_train, axis=0)
        self.x_min = np.min(X_train, axis=0)
        self._log.debug(f'\n(x_min, x_max): {(self.x_min, self.x_max)}')
        self._log.debug(f'\nx_min.shape: {self.x_min.shape}')
        self.X_train = self._normalize(X_train)
        self._log.debug(f'\nX_train.shape: {self.X_train.shape}')
        self.y_train = y_train
        self._log.debug(f'\nunique_labels: {np.unique(y_train)}')
        return self

    def _predict_non_vectorized(self, X):
        y_predict = []
        for x in X:
            # Normalize input:
            x = self._normalize(x)
            # Calculate differences:
            diff = self.X_train - x
            # Calculate the distances using a specified norm along the columns!
            dist = np.linalg.norm(diff, axis=1, ord=self.norm_ord)
            # argpartition puts the k-th entry at the correct position, all smaller are left of that (those are the k
            # nearest neighbors!), all larger are on the right (both without caring for the order there):
            knn_idx = np.argpartition(dist, self.k)[0:self.k]
            # get the labels of the k nearest neighbors:
            knn_labels = self.y_train[knn_idx]
            # Count how many of those belong to which classification:
            classification, counts = np.unique(knn_labels, return_counts=True)
            y_predict.append(classification[np.argmax(counts)])
        return np.asarray(y_predict)

    def _predict_vectorized(self, X):
        # Normalize input: X.shape = (r, f) with r: # of records, f: # of features
        X = self._normalize(X)
        # Calculate differences: needs reshaping to (1, t, f) and (r, 1, f) with t: # of training records
        diff = self.X_train[None, :, :] - X[:, None, :]  # result has shape (r, t, f)
        self._log.debug(f'\ndiff.shape: {diff.shape}')
        # Calculate the distances using a specified norm along the columns!
        dist = np.linalg.norm(diff, axis=-1, ord=self.norm_ord)  # result has shape (r, t)
        self._log.debug(f'\ndist.shape: {dist.shape}')
        # argpartition puts the k-th entry at the correct position, all smaller are left of that (those are the k
        # nearest neighbors!), all larger are on the right (both without caring for the order there):
        partition_idx = np.argpartition(dist, self.k, axis=-1)  # result still has shape (r, t) but sorted t's!
        knn_idx = partition_idx[:, :self.k]  # only use the k nearest neighbors (along last axis)! shape: (r, k)
        self._log.debug(f'\nknn_idx.shape: {knn_idx.shape}')
        # get the labels of the k nearest neighbors (knn_idx contains lookup indices for y_train/X_train rows!)
        knn_labels = self.y_train[knn_idx]  # 1D-vector (t,) indexed with 2D-vector (r, k) works! result shape (r, k)
        self._log.debug(f'\nknn_labels.shape: {knn_labels.shape}')
        # Count how many of those belong to which classification:
        y_predict = []
        for knn_label_row in knn_labels:
            classification, counts = np.unique(knn_label_row, return_counts=True)
            y_predict.append(classification[np.argmax(counts)])
        return np.asarray(y_predict)

    def _normalize(self, X):
        return (X - self.x_min) / (self.x_max - self.x_min)


class KNearestNeighborWeightedClassifier(object):

    _log = logging.getLogger(__name__ + '.KNearestNeighborWeightedClassifier')

    def __init__(self, k=4, smear=1, norm_ord=None):
        self.k = k  # Number of nearest neighbors!
        self.norm_ord = norm_ord  # Order of the norm to use!
        self.smear = smear  # Smear value that prevents inverse weights from diverging!

    def fit(self, X_train, y_train):
        # Normalize data (over all samples/rows) and save stuff:
        self.x_max = np.max(X_train, axis=0)
        self.x_min = np.min(X_train, axis=0)
        self.X_train = self._normalize(X_train)
        self.tree = KDTree(self.X_train)
        self.y_train = y_train

    def predict(self, X):
        # Normalize input: X.shape = (r, f) with r: # of records, f: # of features
        X = self._normalize(X)

        dists, neighbours = self.tree.query(X, self.k)  # Use the KDTree to find the k nearest neighbours and distances!
        self._log.debug(f'\ndists.shape: {dists.shape}')
        self._log.debug(f'\nneighbours.shape: {neighbours.shape}')
        dists += self.smear/self.k  # Add distance offset (avoids division by zero), divided equally among k neighbours.
        weights = 1 / dists  # weights are inverse to distances!
        self._log.debug(f'\nweights.shape: {weights.shape}')
        weight_sum_vec = weights.sum(axis=1)
        weight_sum_arr = np.repeat(weight_sum_vec[:, None], self.k, axis=1)
        self._log.debug(f'\nweight_sum_arr.shape: {weight_sum_arr.shape}')
        weights_normed = weights / weight_sum_arr
        self._log.debug(f'\nweights_normed.shape: {weights_normed.shape}')
        categories = (weights_normed * self.y_train[neighbours]).sum(axis=-1)
        y = np.round(categories, decimals=0).astype(int)
        self._log.debug(f'\ny.shape: {y.shape}')
        return y

    def _normalize(self, X):
        return (X - self.x_min) / (self.x_max - self.x_min)
