"""Task 2 student exercise: OLS and coefficient of determination."""
import numpy as np


def fit_normal_equation(Xb, y):
    gram = Xb.T @ Xb
    rhs = Xb.T @ y
    return np.linalg.solve(gram, rhs)


def r2_score(y, yhat):
    residual = y - yhat
    centered = y - y.mean()
    return float(1 - (residual @ residual) / (centered @ centered))
