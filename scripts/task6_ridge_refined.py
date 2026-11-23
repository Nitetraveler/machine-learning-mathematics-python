"""Supplementary Ridge model selection using training-only CV.

Author: Tariq Hussain
Original reference experiment retained unchanged.
"""
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from mlmath.linreg import add_bias, mse
from mlmath.model_selection import (
    polynomial_features,
    ridge_fit,
    cross_val_mse,
)

rng = np.random.default_rng(3)

x = rng.uniform(-1, 1, 40)
y = np.sin(np.pi*x) + rng.normal(0, 0.3, 40)

P = polynomial_features(x, 15)

# Refine the original logarithmic grid.
# Model selection uses training folds only.
lams = np.logspace(-6, 1, 141)

cv_scores = np.array([
    cross_val_mse(P, y, lam, k=5, seed=0)
    for lam in lams
])

idx = int(np.argmin(cv_scores))
best_lam = float(lams[idx])
best_cv = float(cv_scores[idx])

print("========== TRAINING-ONLY MODEL SELECTION ==========")
print("Lambda candidates:", len(lams))
print("Selected lambda:", best_lam)
print("Best CV MSE:", best_cv)

w = ridge_fit(add_bias(P), y, best_lam)

# Evaluate only after selecting lambda.
xt = rng.uniform(-1, 1, 2000)
yt = np.sin(np.pi*xt) + rng.normal(0, 0.3, 2000)

Pt = polynomial_features(xt, 15)

pred = add_bias(Pt) @ w
test_mse = mse(yt, pred)

sk = Ridge(alpha=best_lam).fit(P, y)
sk_pred = sk.predict(Pt)
sk_mse = mean_squared_error(yt, sk_pred)

diff = float(np.max(np.abs(pred - sk_pred)))

np.testing.assert_allclose(
    pred, sk_pred, atol=1e-8
)

with open("outputs/task6_ridge.json") as f:
    original = json.load(f)

result = {
    "author": "Tariq Hussain",
    "selection_method": "training-only 5-fold CV",
    "lambda_candidates": len(lams),
    "best_lambda": best_lam,
    "best_cv_mse": best_cv,
    "test_mse": float(test_mse),
    "sklearn_test_mse": float(sk_mse),
    "max_prediction_difference": diff,
    "original_test_mse": original["test_mse_best"],
    "threshold_satisfied": bool(test_mse < 0.15),
}

with open("outputs/task6_ridge_refined.json", "w") as f:
    json.dump(result, f, indent=2)

print()
print("========== REFINED MODEL RESULTS ==========")
print("Original test MSE:", original["test_mse_best"])
print("Refined test MSE:", test_mse)
print("sklearn test MSE:", sk_mse)
print("Maximum prediction difference:", diff)
print("Threshold satisfied:", test_mse < 0.15)

print("PASS: Refined training-only CV experiment")
