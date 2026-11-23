"""Task 7 guided PCA implementation exercise.

Author: Tariq Hussain
Implementation completed with AI assistance.
"""
import numpy as np


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_

        _, singular_values, directions = np.linalg.svd(
            centered, full_matrices=False
        )

        variance = singular_values**2 / (len(X) - 1)

        k = (
            len(singular_values)
            if self.n_components is None
            else self.n_components
        )

        self.explained_variance_all_ = variance
        self.explained_variance_ = variance[:k]
        self.explained_variance_ratio_ = variance[:k] / variance.sum()
        self.components_ = directions[:k]
        self.n_components_ = k

        return self


def n_components_for_variance(ratio, target):
    cumulative = np.cumsum(ratio)
    return int(np.searchsorted(cumulative, target) + 1)
