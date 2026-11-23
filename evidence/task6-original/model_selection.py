"""Polynomial features, ridge regression and k-fold cross-validation."""
import numpy as np
from mlmath.linreg import add_bias, mse


def polynomial_features(x, degree):
    x = np.asarray(x, dtype=float).ravel()
    return np.column_stack([x ** k for k in range(1, degree + 1)])


def ridge_fit(Xb, y, lam):
    Xb = np.asarray(Xb, float)
    y = np.asarray(y, float)
    P = lam * np.eye(Xb.shape[1])
    P[0, 0] = 0.0
    return np.linalg.solve(Xb.T @ Xb + P, Xb.T @ y)


def kfold_indices(n, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, folds[i]


def cross_val_mse(X, y, lam, k=5, seed=0):
    scores = []
    for tr, va in kfold_indices(len(y), k, seed):
        w = ridge_fit(add_bias(X[tr]), y[tr], lam)
        scores.append(mse(y[va], add_bias(X[va]) @ w))
    return float(np.mean(scores))
