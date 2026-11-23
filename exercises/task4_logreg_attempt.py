"""Task 4 guided exercise correction after initial failed attempt."""
import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def loss_and_grad(Xb, y, w, l2=0.0):
    y = np.asarray(y, dtype=float)
    p = sigmoid(Xb @ w)
    q = np.clip(p, 1e-15, 1 - 1e-15)

    loss = -np.mean(
        y * np.log(q) + (1 - y) * np.log(1 - q)
    )

    penalty = w.copy()
    penalty[0] = 0

    loss += 0.5 * l2 * np.dot(penalty, penalty)
    grad = Xb.T @ (p - y) / len(y) + l2 * penalty

    return float(loss), grad
