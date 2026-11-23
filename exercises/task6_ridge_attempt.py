"""Task 6 guided exercise correction."""
import numpy as np


def ridge_fit(Xb, y, lam):
    Xb = np.asarray(Xb, dtype=float)
    y = np.asarray(y, dtype=float)

    penalty = np.eye(Xb.shape[1])
    penalty[0, 0] = 0.0

    A = Xb.T @ Xb + lam * penalty
    b = Xb.T @ y

    return np.linalg.solve(A, b)


def kfold_indices(n, k=5, seed=0):
    indices = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(indices, k)

    for i in range(k):
        validation = folds[i]
        training = np.concatenate(
            [fold for j, fold in enumerate(folds) if j != i]
        )
        yield training, validation
