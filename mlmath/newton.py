"""Newton/IRLS logistic regression.

Author: Tariq Hussain
AI-assisted challenge implementation.
"""
import numpy as np
from mlmath.logreg import sigmoid, loss_and_grad


def fit_newton(Xb, y, l2=0.0, max_iter=50, tol=1e-12):
    Xb = np.asarray(Xb, dtype=float)
    y = np.asarray(y, dtype=float)

    n, d = Xb.shape
    w = np.zeros(d)

    P = np.eye(d)
    P[0, 0] = 0.0

    # At 100,000 features a dense float64 Hessian requires
    # approximately 80 GB of memory, excluding temporary arrays.
    # Forming H costs O(n*d^2); solving costs O(d^3).
    # This makes dense Newton impractical at that scale.

    for iteration in range(1, max_iter + 1):
        _, grad = loss_and_grad(Xb, y, w, l2=l2)

        p = sigmoid(Xb @ w)
        weights = p * (1.0 - p)

        # Weighted multiplication avoids constructing dense W.
        H = (Xb.T @ (weights[:, None] * Xb)) / n
        H += l2 * P

        step = np.linalg.solve(H, grad)

        print(
            f"Iteration {iteration:2d}: "
            f"||grad||={np.linalg.norm(grad):.12e} "
            f"||step||={np.linalg.norm(step):.12e}"
        )

        w -= step

        if np.linalg.norm(step) < tol:
            break

    return w, iteration
